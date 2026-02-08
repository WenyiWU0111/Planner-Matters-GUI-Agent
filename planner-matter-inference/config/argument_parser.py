"""Argument parser configuration for the GUI Agent"""
import argparse


def config() -> argparse.Namespace:
    """Configure and parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation on the benchmark"
    )

    # -------------------------------------------------------------------------
    # Evaluation & run
    # -------------------------------------------------------------------------
    parser.add_argument("--evaluation_type", type=str, default="mmina",
                        choices=["mmina", "webvoyager", "mind2web", "mind2web_executable"],
                        help="Benchmark to run: mmina, webvoyager, mind2web, or mind2web_executable")
    parser.add_argument("--domain", type=str, default="wikipedia",
                        help="Domain / site subset (e.g. Amazon, wikipedia)")
    parser.add_argument("--test_start_idx", type=int, default=0,
                        help="Start index for test tasks")
    parser.add_argument("--test_end_idx", type=int, default=10000,
                        help="End index for test tasks")
    parser.add_argument("--result_dir", type=str, default="",
                        help="Output directory for results (default: results/<evaluation_type>/<model>/<domain>/<datetime>)")
    parser.add_argument("--datetime", type=str, default=None,
                        help="Run identifier; timestamp is appended if set")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Enable debug mode")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to the data directory")

    # -------------------------------------------------------------------------
    # Browser environment
    # -------------------------------------------------------------------------
    parser.add_argument("--render", action="store_true",
                        help="Render the browser window")
    parser.add_argument("--slow_mo", type=int, default=0,
                        help="Slow down browser actions by this amount (ms)")
    parser.add_argument("--observation_type",
                        choices=["accessibility_tree", "html", "image"],
                        default="image",
                        help="Observation type for the agent")
    parser.add_argument("--current_viewport_only", action="store_true",
                        help="Only use the current viewport for observation")
    parser.add_argument("--viewport_width", type=int, default=1280)
    parser.add_argument("--viewport_height", type=int, default=720)
    parser.add_argument("--save_trace_enabled", action="store_true",
                        help="Save browser trace for debugging")
    parser.add_argument("--sleep_after_execution", type=float, default=0.0)
    parser.add_argument("--max_steps", type=int, default=15,
                        help="Maximum steps per task before early stop")
    parser.add_argument("--imgbin_dir", type=str, default="")  # Not in use

    # -------------------------------------------------------------------------
    # Agent behavior
    # -------------------------------------------------------------------------
    parser.add_argument("--agent_type", type=str, default="prompt")
    parser.add_argument("--parsing_failure_th", type=int, default=3,
                        help="Stop after this many consecutive parsing failures")
    parser.add_argument("--repeating_action_failure_th", type=int, default=2,
                        help="Stop after this many consecutive repeating actions")
    parser.add_argument("--task_cnt", type=int, default=0,
                        help="Task counter (for logging / indexing)")
    parser.add_argument("--hop_cnt", type=int, default=0,
                        help="Hop counter (for multi-hop logging)")

    # -------------------------------------------------------------------------
    # Main model (actor)
    # -------------------------------------------------------------------------
    parser.add_argument("--provider", type=str, default="custom")
    parser.add_argument("--model", type=str, default="qwen2.5-vl",
                        help="Main VLM for the actor (e.g. qwen2.5-vl, gpt-4o)")
    parser.add_argument("--checkpoint_path", type=str,
                        default="WenyiWU0111/Qwen2_5_7B_RL_baseline",
                        help="HF repo id or local path for continuous-memory (QFormer) checkpoint")
    parser.add_argument("--loaded_tokenizer", default=None)
    parser.add_argument("--loaded_model", default=None)
    parser.add_argument("--mode", type=str, default="chat")
    parser.add_argument("--context_length", type=int, default=0)

    # -------------------------------------------------------------------------
    # Planner
    # -------------------------------------------------------------------------
    parser.add_argument("--use_planner", action="store_true", default=False,
                        help="Use a planner for task decomposition")
    parser.add_argument("--use_planner_with_memory", action="store_true", default=False,
                        help="Use planner with experience memory for task planning")
    parser.add_argument("--planner_server_url", type=str, default="http://localhost:8000/v1",
                        help="vLLM / API server URL for the planner")
    parser.add_argument("--planner_model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help="Model name for the planner")
    parser.add_argument("--checkpoint_planner", type=str, default=None,
                        help="Local checkpoint path for the planner model")
    parser.add_argument("--use_base_planner_model", action="store_true", default=False,
                        help="Use base planner model instead of fine-tuned")
    parser.add_argument("--use_history_context", action="store_true", default=False,
                        help="Feed action success/failure context into planner updates")
    parser.add_argument("--use_adaptive_memory", action="store_true", default=False,
                        help="Let VLM refine memory query (adaptive focus) during execution")

    # -------------------------------------------------------------------------
    # Grounding & tool model
    # -------------------------------------------------------------------------
    parser.add_argument("--grounding_model_name", type=str, default="ui-ins-7b",
                        help="Model for element grounding (pixel / accessibility)")
    parser.add_argument(
        "--grounding_mode",
        type=str,
        default="auto",
        choices=["auto", "prefer", "force", "off"],
        help="How to use grounding: auto (coords→id→grounding), prefer (coords→grounding→id), "
             "force (always grounding), off (no grounding).",
    )
    parser.add_argument("--tool_model_name", type=str, default="qwen2.5-vl",
                        help="Model name for tool / helper LLM")

    # -------------------------------------------------------------------------
    # Generation
    # -------------------------------------------------------------------------
    parser.add_argument("--max_tokens", type=int, default=500,
                        help="Max tokens per model response")
    parser.add_argument("--stop_token", type=str, default=None)

    # -------------------------------------------------------------------------
    # Memory (continuous / discrete)
    # -------------------------------------------------------------------------
    parser.add_argument("--use_history", type=bool, default=False)
    parser.add_argument("--use_continuous_memory", type=bool, default=False,
                        help="Use continuous (e.g. QFormer) memory")
    parser.add_argument("--use_discrete_memory", type=bool, default=False,
                        help="Retrieve similar trajectories, summarize with VLM, inject into prompt")
    parser.add_argument("--discrete_memory_cache_path", type=str,
                        default="data/test_hybrid/trajectory_summaries.json",
                        help="JSON cache for discrete memory summaries (by file_id)")
    parser.add_argument("--discrete_memory_max_actions", type=int, default=8,
                        help="Max actions per trajectory for summarization")
    parser.add_argument("--discrete_memory_use_checkpoint", type=bool, default=False,
                        help="Use main LLM checkpoint for discrete summarization (instead of tool_llm)")
    parser.add_argument("--faiss_index_path", type=str, default="memory_index/multimodal_549",
                        help="FAISS index path for memory retrieval (prefix, no extension)")
    parser.add_argument("--similar_num", type=int, default=10,
                        help="Number of similar trajectories to retrieve")
    parser.add_argument("--bank_size", type=int, default=None)
    parser.add_argument("--memory_data_dir", type=str, nargs="+", default=["data/trajectories"],
                        help="One or more dirs containing trajectory data for memory")
    parser.add_argument("--max_obs_length", type=int, default=1920,
                        help="Truncate observation to this length (0 = no truncation)")

    # -------------------------------------------------------------------------
    # Reasoning bank
    # -------------------------------------------------------------------------
    parser.add_argument("--use_reasoning_bank", type=bool, default=False,
                        help="Inject retrieved reasoning items into the agent")
    parser.add_argument("--reasoning_bank_path", type=str, default="memory/reasoning_bank.jsonl",
                        help="Path to reasoning bank JSONL")
    parser.add_argument("--reasoning_top_k", type=int, default=2,
                        help="Number of reasoning items to inject at first turn")
    parser.add_argument("--reasoning_domain_filter", type=bool, default=True,
                        help="Filter reasoning items by current domain")
    parser.add_argument("--reasoning_index_base", type=str, default="memory_index/reasoning_bank_text",
                        help="Base path for reasoning bank FAISS index")
    parser.add_argument("--reasoning_bank_multimodal", type=bool, default=False,
                        help="Use multimodal reasoning bank (key steps + screenshots)")

    # -------------------------------------------------------------------------
    # Other memory / workflow
    # -------------------------------------------------------------------------
    parser.add_argument("--use_awm", action="store_true", default=False,
                        help="Enable AWM (Agent Workflow Memory)")

    # -------------------------------------------------------------------------
    # Training data collection
    # -------------------------------------------------------------------------
    parser.add_argument("--collect_training_data", action="store_true", default=False,
                        help="Collect prompts and responses for training")
    parser.add_argument("--training_data_dir", type=str, default="training_data",
                        help="Directory to save collected training data")
    parser.add_argument("--save_examples_memory", action="store_true", default=False,
                        help="Add example memory to the agent when collecting")

    # -------------------------------------------------------------------------
    # Evaluation output
    # -------------------------------------------------------------------------
    parser.add_argument("--render_screenshot", action="store_true",
                        help="Render screenshots during evaluation")

    # -------------------------------------------------------------------------
    # Parse and post-process
    # -------------------------------------------------------------------------
    args = parser.parse_args()

    # Instruction prompt path (used by agent)
    args.instruction_path = "agent/prompts/jsons/p_cot_ground_actree_2s.json"

    # Continuous memory: align model name with QFormer
    if args.use_continuous_memory and "full-sft" not in args.model:
        args.model = "agent-qformer_" + args.model

    # Unique run id
    from datetime import datetime as dt
    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
    if args.datetime is None:
        args.datetime = timestamp
    else:
        args.datetime = f"{args.datetime}_{timestamp}"

    # Default result and training dirs
    if not args.result_dir:
        args.result_dir = f"results/{args.evaluation_type}/{args.model}/{args.domain}/{args.datetime}"
    args.training_data_dir = f"training_data/{args.evaluation_type}/{args.domain}/{args.model}/{args.datetime}"

    return args
