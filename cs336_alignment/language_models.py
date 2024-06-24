from vllm import LLM, SamplingParams

MODELPATH_8B = "/data/Meta-Llama-3-8B"
MODELPATH_80B_INSTRUCT = "/home/shared/Meta-Llama-3-70B-Instruct"


def get_prompt(instruction):
    return (
        "# Instruction\n"
        "Below is a list of conversations between a human and an AI assistant (you).\n"
        "Users place their queries under \"# Query:\", and your responses are under \"# Answer:\".\n"
        "You are a helpful, respectful, and honest assistant.\n"
        "You should always answer as helpfully as possible while ensuring safety.\n"
        "Your answers should be well-structured and provide detailed information. They should also have an engaging tone.\n"
        "Your responses must not contain any fake, harmful, unethical, racist, sexist, toxic, dangerous, or illegal content, even if it may be helpful.\n"
        "Your response must be socially responsible, and thus you can reject to answer some controversial topics.\n"
        "\n"
        "# Query:\n"
        f"```{instruction}```\n"
        "\n"
        "# Answer:\n"
        "```"
    )

def sample_from_model(model, prompts, temperature=0.0, top_p=1.0, max_tokens=1024, stop=["\n"]):
    sampling_params = SamplingParams(
        temperature=temperature, 
        top_p=top_p, 
        max_tokens=max_tokens, 
        stop=stop
    )
    outputs = model.generate(prompts, sampling_params)
    return outputs

def load_model(model_path):
    llm = LLM(model=model_path)
    return llm
