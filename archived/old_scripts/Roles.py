from gpt4all import GPT4All
import GPUtil
from time import sleep, time
import threading
import multiprocessing
from pathlib import Path
import sys
import json


# Define the maximum number of conversation entries before summarizing
MAX_ENTRIES_PER_MODEL = 5  # Adjust as needed

# Define paths for conversation and summary JSON files
conversation_output_path = Path.cwd() / "model_results.json"
summary_output_path = Path.cwd() / "model_summary.json"

# Initialize the global monitoring flag

def monitor_gpu():
    """Thread function to monitor GPU status in real-time."""
    global monitoring
    while monitoring:
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                print(f"GPU ID: {gpu.id}", flush=True)
                print(f"Name: {gpu.name}", flush=True)
                print(f"Load: {gpu.load * 100:.2f}%", flush=True)
                print(f"Memory Free: {gpu.memoryFree:.2f} MB", flush=True)
                print(f"Memory Used: {gpu.memoryUsed:.2f} MB", flush=True)
                print(f"Memory Total: {gpu.memoryTotal:.2f} MB", flush=True)
                print(f"Temperature: {gpu.temperature} °C", flush=True)
                print("-" * 30, flush=True)
        except Exception as e:
            print(f"Error monitoring GPU: {e}", flush=True)
        sleep(1)  # Monitor every 1 second

def run_model_task(args):
    """Function to run model inference in a separate process."""
    key, model_name, prompts, result_list = args
    
    # Each process should initialize its own model instance
    system_prompt = generate_system_prompt_from_history(key)
    try:
        model = GPT4All(model_name, device='gpu')
    except Exception as e:
        print(f"[{key}] Failed to initialize model '{model_name}': {e}", flush=True)
        for prompt in prompts:
            result_list.append((key, prompt, None))
        return

    try:
        with model.chat_session(system_prompt=system_prompt):
            for prompt in prompts:
                tokens = ""
                try:
                    for token in model.generate(prompt, streaming=True, max_tokens=50, temp = 0.2, top_k = 40, top_p = 0.4, repeat_penalty = 1.18, repeat_last_n = 64):
                        tokens += token
                except Exception as e:
                    print(f"[{key}] Error during generating response for prompt '{prompt}': {e}", flush=True)
                    tokens = None
                # Append the response
                result_list.append((key, prompt, tokens))
    except Exception as e:
        print(f"[{key}] Error during chat session: {e}", flush=True)
        for prompt in prompts:
            result_list.append((key, prompt, None))

def generate_system_prompt_from_history(model_key: str) -> str:
    """
    Generate a system_prompt based on whether a summary file or conversation logs exist.
    - If a summary file exists and has content, it will be used first.
    - Otherwise, attempt to load conversation logs (only the part corresponding to model_key).
    - If neither is available, use a default opening system prompt.
    """
    loaded_summary = None

    # 1) Try to load a summary
    if 'model' not in locals():
        if summary_output_path.exists():
            try:
                with open(summary_output_path, 'r', encoding='utf-8') as f:
                    summaries = json.load(f)
                    # Usually the summary file is a list, each with a "summary" field
                    if isinstance(summaries, list) and len(summaries) > 0:
                        # Take the last summary as the conversation context for the specific model
                        # Alternatively, filter summaries by model_key if multiple models are summarized
                        for summary in reversed(summaries):
                            if summary.get("model_used") == model_key:
                                loaded_summary = summary.get("summary", "")
                                break
            except Exception as e:
                print(f"Error loading summary from '{summary_output_path}': {e}", flush=True)

        # 2) If a summary is loaded, build system_prompt from it
        if loaded_summary:
            system_prompt = (
                "Below is a summary of our previous conversation:"
                f"{loaded_summary}"
                "answer with one sentence"
            )
            return system_prompt
        
        # 3) If there's no summary, fall back to reading conversation logs from model_results.json
        elif conversation_output_path.exists():
            try:
                with open(conversation_output_path, 'r', encoding='utf-8') as f:
                    conv_data = json.load(f)  # Expected format: { "a": {"conversation": [...]}, "b": {...}, ...}
        
                if isinstance(conv_data, dict) and model_key in conv_data:
                    conversation_list = conv_data[model_key].get("conversation", [])
                    # Concatenate the most recent conversation lines into a simple text format
                    logs = ""
                    for entry in conversation_list:
                        user_text = entry.get("prompt", "")
                        model_text = entry.get("response", "")
                        logs += f"I: {user_text}\n You: {model_text}\n"
                    
                    if logs.strip():
                        return (
                            "Below is our previous conversation:"
                            f"{logs}"
                            "answer with one sentence"
                        )
            except Exception as e:
                print(f"Error loading conversation logs from '{conversation_output_path}': {e}", flush=True)

        # 4) If there is no summary and no conversation logs, use a default system_prompt
    else:
        return "answer with one sentence"

def load_json(json_path):
    """Load JSON data from a file. Returns an empty dict if file doesn't exist or is invalid."""
    if json_path.exists():
        with open(json_path, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print(f"Error: '{json_path}' is not a valid JSON file.", flush=True)
                return {}
    else:
        return {}

def save_json(data, json_path):
    """Save data to a JSON file with proper formatting."""
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Error saving JSON to '{json_path}': {e}", flush=True)

def summarize_conversation(args):
    """
    Summarize the given conversation using the specified model and overwrite existing summary.

    Parameters:
    - args (tuple): (model_name, conversation)

    Returns:
    - dict: A dictionary containing "timestamp", "model_used", and "summary".
    """
    model_name, conversation = args

    # Use the last n entries from the conversation for summarization
    conversation_subset = conversation[-MAX_ENTRIES_PER_MODEL:] if len(conversation) >= MAX_ENTRIES_PER_MODEL else conversation

    try:
        # Initialize the summarization model
        model = GPT4All(model_name, device='gpu')
    except Exception as e:
        print(f"[summarize] Failed to initialize model '{model_name}': {e}", flush=True)
        return None

    # Prepare the summarization prompt
    summary_prompt = "Make a summary:"
    for entry in conversation_subset:
        prompt = entry.get("prompt", "")
        response = entry.get("response", "")
        summary_prompt += f"me: {prompt}\n you: {response}\n"

    # Generate the summary
    summary_text = ""
    try:
        for token in model.generate(summary_prompt, streaming=True, max_tokens=150, temp = 0.2, top_k = 40, top_p = 0.4, repeat_penalty = 1.18, repeat_last_n = 64):
            summary_text += token
    except Exception as e:
        print(f"[summarize] Error during generating summary: {e}", flush=True)
        summary_text = None

    # Create the new summary dictionary
    new_summary = {
        "timestamp": int(time()),
        "model_used": model_name,
        "summary": summary_text
    }

    return new_summary

def main_task(models_prompts, GPU_monitoring=True):
    """Main function to execute the model tasks."""
    global monitoring
    monitoring = GPU_monitoring  # Set the monitoring flag

    # Start GPU monitoring thread if monitoring is enabled
    if monitoring:
        gpu_monitor_thread = threading.Thread(target=monitor_gpu, daemon=True)
        gpu_monitor_thread.start()
        print("GPU monitoring started.", flush=True)
    
    # Define the four independent small models and their prompts
    models_prompts = models_prompts

    # Load existing conversations
    conversations = load_json(conversation_output_path)
    
    # Initialize conversation entries for each model if not present
    for key in models_prompts.keys():
        if key not in conversations:
            conversations[key] = {"conversation": []}
    
    # Define the default path where models are stored
    default_model_path = Path.home() / ".cache/gpt4all"
    
    # Ensure all model files exist
    missing_models = []
    for key, info in models_prompts.items():
        model_file = default_model_path / info['model_name']
        if not model_file.exists():
            missing_models.append(info['model_name'])
            print(f"Model file not found for key '{key}': {model_file}", flush=True)
    
    if missing_models:
        print("\nMissing models:")
        for m in missing_models:
            print(f" - {m}", flush=True)
        print("\nPlease ensure all models are downloaded before running the script.", flush=True)
        sys.exit(1)

    # Use multiprocessing manager to create a shared list
    with multiprocessing.Manager() as manager:
        result_list = manager.list()

        # Create Q&A tasks for each model
        tasks = []
        for key, info in models_prompts.items():
            model_name = info['model_name']
            prompts = info['prompts']
            tasks.append((key, model_name, prompts, result_list))

        start_time = time()  # Start timing

        try:
            # Run Q&A tasks using multiprocessing
            pool_size = 4  # Adjust based on GPU's capability
            with multiprocessing.Pool(processes=pool_size) as pool:
                pool.map(run_model_task, tasks)
        finally:
            # Step 3: Stop GPU monitoring
            monitoring = False
            gpu_monitor_thread.join()
            print("GPU monitoring stopped.", flush=True)

        end_time = time()  # End timing
        print(f"\nTotal Time Taken: {end_time - start_time:.2f} seconds", flush=True)
        
        # Append the generated Q&A to conversations
        for entry in result_list:
            key, prompt, response = entry
            conversations[key]['conversation'].append({
                "prompt": prompt,
                "response": response
            })
        
        # Save the updated conversations to the JSON file
        try:
            save_json(conversations, conversation_output_path)
            print(f"\nResults have been saved to '{conversation_output_path}'.", flush=True)
        except Exception as e:
            print(f"Failed to save results to JSON: {e}", flush=True)
        
        # Display all Q&A results
        print("\nAll Results:")
        for idx, (key, prompt, response) in enumerate(result_list):
            print(f"Model '{key}' - Prompt: {prompt}", flush=True)
            print(f"  Generated Response: {response}", flush=True)
            print("-" * 50, flush=True)    

    # Identify models that need summarization
    summarization_tasks = []
    for key, info in models_prompts.items():
        model_conversation = conversations[key]['conversation']
        if len(model_conversation) >= MAX_ENTRIES_PER_MODEL:
            print(f"\nModel '{key}' conversation exceeds {MAX_ENTRIES_PER_MODEL} entries. Preparing to summarize...", flush=True)
            summarization_tasks.append((info['model_name'], model_conversation))
    
    # Run summarization tasks in parallel
    if summarization_tasks:
        print(f"\nStarting parallel summarization for {len(summarization_tasks)} models...", flush=True)
        with multiprocessing.Pool(processes=4) as pool:  # Adjust pool size as needed
            summaries = pool.map(summarize_conversation, summarization_tasks)
    
        # Filter out any None summaries due to errors
        summaries = [s for s in summaries if s is not None]
    
        if summaries:
            # Load existing summaries
            existing_summaries = load_json(summary_output_path)
            if not isinstance(existing_summaries, list):
                print("Summary JSON file format error. Initializing empty summary.", flush=True)
                existing_summaries = []
    
            # Update summaries: ensure each model has only one summary
            for summary in summaries:
                # Remove old summary for the same model
                existing_summaries = [s for s in existing_summaries if s.get("model_used") != summary["model_used"]]
                # Add the new summary
                existing_summaries.append(summary)
    
            # Save the updated summaries
            try:
                save_json(existing_summaries, summary_output_path)
                print(f"\nSummaries have been saved to '{summary_output_path}'.", flush=True)
            except Exception as e:
                print(f"[summarize] Failed to save summaries to JSON: {e}", flush=True)
    
        # Optionally: Display all summary results
        print("\nAll Summary Results:")
        if summary_output_path.exists():
            summaries = load_json(summary_output_path)
            for summary_entry in summaries:
                print(f"Timestamp: {summary_entry['timestamp']}", flush=True)
                print(f"Model Used: {summary_entry['model_used']}", flush=True)
                print(f"Summary: {summary_entry['summary']}", flush=True)
                print("-" * 50, flush=True)
        else:
            print("No summaries found.", flush=True)
    
# Main entry point
if __name__ == "__main__":
    multi_prompts={
        'a': {
            'model_name': 'Meta-Llama-3-8B-Instruct.Q4_0_a.gguf',
            'prompts': [
                "Remember my Likes and dislikes about Paris?",
            ]
        },
        'b': {
            'model_name': 'Meta-Llama-3-8B-Instruct.Q4_0_a.gguf',
            'prompts': [
                "What is the closest species to human?",
            ]
        },
        'c': {
            'model_name': 'Meta-Llama-3-8B-Instruct.Q4_0_a.gguf',
            'prompts': [
                "What is the closest species to human?",
            ]
        },
        'd': {
            'model_name': 'Meta-Llama-3-8B-Instruct.Q4_0_a.gguf',
            'prompts': [
                "There are how many 'r' in 'strawberry'?",
            ]
        }
    }

    single_prompts={
        'a': {
            'model_name': 'Meta-Llama-3-8B-Instruct.Q4_0_a.gguf',
            'prompts': [
                "Hi, my name's Raven",
                "My favorite food is noodles",
                "I like Paris, but I don't like Eiffel Tower",
                "do you think I am weird?",
                "Thanks, nice talking to you today"
            ]
        },
    }
    
    back_prompts={
        'a': {
            'model_name': 'Meta-Llama-3-8B-Instruct.Q4_0_a.gguf',
            'prompts': [
                "What did we talk last time? Do you remember my favorite food?"
            ]
        },
    }



    #main_task(models_prompts = single_prompts , GPU_monitoring = True)
    main_task(models_prompts = back_prompts , GPU_monitoring = True)