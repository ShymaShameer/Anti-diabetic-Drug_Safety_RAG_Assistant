## for ignoring the langchain module import error
import sys
import types

# Bypass unused VertexAI import check in Ragas
dummy_module = types.ModuleType("langchain_community.chat_models.vertexai")
dummy_module.ChatVertexAI = type("ChatVertexAI", (object,), {})
sys.modules["langchain_community.chat_models.vertexai"] = dummy_module


from evaluation.evaluation_engine import (
    # evaluate_baseline,
    # evaluate_chunking_strategies,
    # evaluate_reranker_strategies,
    evaluate_query_rewriting,
)


if __name__ == "__main__":

    # Run Stage 1: Baseline Evaluation
    # evaluate_baseline()

    # Run Stage 2: Chunking Strategy Evaluation
    # evaluate_chunking_strategies()

    # Run Stage 3: Reranker Strategy Evaluation
    # evaluate_reranker_strategies()

    # Run Stage 4: Query Rewriter Evaluation
    evaluate_query_rewriting()