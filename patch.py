# Feature 2: SSE
f2 = open("backend/api/main.py", "r")
content = f2.read()
f2.close()

sse = '''
@app.get("/stream", tags=["Research"])
async def stream_query(question: str = Query(...)):
    async def event_generator():
        import json
        try:
            response = await run_research_query(question=question)
            for step in response.agent_trace:
                yield f"data: " + json.dumps({"step": step.agent, "action": step.action}) + "\\n\\n"
            yield f"data: " + json.dumps({"answer": response.answer, "confidence": response.confidence, "done": True}) + "\\n\\n"
        except Exception as e:
            yield f"data: " + json.dumps({"error": str(e)}) + "\\n\\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream")
'''

content = content.replace("# \u2500\u2500 Exception Handlers", sse + "\n# \u2500\u2500 Exception Handlers")
open("backend/api/main.py", "w").write(content)
print("Feature 2 done")

# Feature 3: BM25
f3 = open("backend/agents/graph_agent.py", "r")
ag = f3.read()
f3.close()

bm25_import = "from rank_bm25 import BM25Okapi\n"
if "rank_bm25" not in ag:
    ag = ag.replace("from config import get_settings", bm25_import + "from config import get_settings")

bm25_code = '''
            # BM25 hybrid reranking
            if vector_results:
                all_docs = [r.content for r in vector_results]
                tokenized = [d.split() for d in all_docs]
                bm25 = BM25Okapi(tokenized)
                bm25_scores = bm25.get_scores(state["question"].split())
                top_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:3]
                context_parts.append("\\n=== BM25 RERANKED TOP RESULTS ===")
                for idx in top_idx:
                    context_parts.append(f"[BM25] {all_docs[idx][:300]}")
'''

ag = ag.replace('        except Exception as e:\n            logger.error("researcher.vector_error"', bm25_code + '        except Exception as e:\n            logger.error("researcher.vector_error"')
open("backend/agents/graph_agent.py", "w").write(ag)
print("Feature 3 done")

# Feature 4: MemoryAgent
memory_node = '''
async def memory_agent_node(state: AgentState) -> dict:
    step_num = len(state["agent_trace"]) + 1
    try:
        cypher = """
        MERGE (m:Memory {query_id: $query_id})
        SET m.question = $question,
            m.answer = $answer,
            m.confidence = $confidence,
            m.created_at = datetime()
        """
        async with neo4j_client._driver.session() as session:
            await session.run(cypher,
                query_id=state["query_id"],
                question=state["question"],
                answer=state["answer"][:500],
                confidence=state["confidence"]
            )
    except Exception as e:
        logger.warning("memory_agent.failed", error=str(e))
    trace_step = AgentStep(
        step=step_num, agent="MemoryAgent",
        action=f"Stored query in Neo4j memory"
    )
    return {"agent_trace": [trace_step]}
'''

ag = open("backend/agents/graph_agent.py").read()
ag = ag.replace("def should_retry", memory_node + "\ndef should_retry")

ag = ag.replace(
    '    graph.add_node("critic", critic_node)',
    '    graph.add_node("critic", critic_node)\n    graph.add_node("memory", memory_agent_node)'
)
ag = ag.replace(
    '{"researcher": "researcher", "end": END}',
    '{"researcher": "researcher", "end": "memory"}'
)
ag = ag.replace(
    '    return graph.compile()',
    '    graph.add_edge("memory", END)\n    return graph.compile()'
)
open("backend/agents/graph_agent.py", "w").write(ag)
print("Feature 4 done")