import time
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

def load_text_data(file_path):
    """
    Load text data from a file.

    Args:
        file_path (str): The path to the text file.

    Returns:
        list: List of strings containing the loaded text data.
    """
    text = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            text.append(line.strip())
    return text

def initialize_base_model(model_id, quantization_config):
    """
    Initialize the base language model for completion.

    Args:
        model_id (str): The base model ID.
        quantization_config (BitsAndBytesConfig): Configuration for model quantization.

    Returns:
        AutoModelForCausalLM: The initialized base language model.
    """
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        use_auth_token=True
    )
    return base_model

def generate_llm_completion(text_prompt, base_model, tokenizer):
    """
    Generate language model completions based on the given text prompt.

    Args:
        text_prompt (str): The input text prompt for language model completion.
        base_model (AutoModelForCausalLM): The base language model for completion.
        tokenizer (AutoTokenizer): The tokenizer associated with the language model.

    Returns:
        str: The generated language model completion for the given prompt.
    """
    base_model.eval()
    model_input = tokenizer(text_prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        result = tokenizer.decode(
            base_model.generate(**model_input, max_new_tokens=150, pad_token_id=2)[0],
            skip_special_tokens=True
        )
    return str(result)

def process_prompt(prompt_type, text, base_model, tokenizer):
    """
    Process a specific prompt type and generate completions.

    Args:
        prompt_type (str): The type of prompt.
        text (list): List of input text data.
        base_model (AutoModelForCausalLM): The base language model for completion.
        tokenizer (AutoTokenizer): The tokenizer associated with the language model.

    Returns:
        None
    """
    start_time = time.time()
    res_file_name = f"test_mistral/{prompt_type}.txt" # Instead of test_mistral, we can put test_starling, test_llama_7b or test_llama2_13b
    with open(res_file_name, "a") as output_file:
        for i in tqdm(range(len(text))):
            prompt = create_prompt(prompt_type, text[i])
            response = generate_llm_completion(prompt, base_model, tokenizer)
            output_file.write(response.strip().replace('\n', '') + '\n')
    
    print(prompt_type)
    print(f"--- {time.time() - start_time} seconds ---")

def create_prompt(prompt_type, text):
    """
    Create a prompt based on the prompt type.

    Args:
        prompt_type (str): The type of prompt.
        text (str): The input text.

    Returns:
        str: The constructed prompt.
    """
    
    # In our experiments, we did zero shot, 7-shots on original LLMs or fine-tuned LLMs, but you can try also these prepared prompts for other few-shot prompting
    if prompt_type == "zero":
        prompt = "Transform the text into a semantic graph, which means, extract the triples from the text in format of lists like the following, [[\"subject\", \"predicate\", \"object\"], [\"subject\", \"predicate\", \"object\"], [\"subject\", \"predicate\", \"object\"]].\nText: " + text + "\nSemantic graph: "
    elif prompt_type == "one":
        prompt = "Transform the text into a semantic graph, which means, extract the triples from the text in format of lists like the following, [[\"subject\", \"predicate\", \"object\"], [\"subject\", \"predicate\", \"object\"], [\"subject\", \"predicate\", \"object\"]].\nExample: \nText: Aarhus Airport serves the city of Aarhus, Denmark.\nSemantic graph: [[\"Aarhus Airport\", \"city Served\", \"Aarhus Denmark\"]].\nText: " + text + "\nSemantic graph:"
    elif prompt_type == "two":
        prompt = "Transform the text into a semantic graph, which means, extract the triples from the text in format of lists like the following, [[\"subject\", \"predicate\", \"object\"], [\"subject\", \"predicate\", \"object\"], [\"subject\", \"predicate\", \"object\"]].\nExample 1: \nText: Aarhus Airport serves the city of Aarhus, Denmark.\nSemantic graph: [[\"Aarhus Airport\", \"city Served\", \"Aarhus Denmark\"]].\nExample 2 :\nText: Aleksandr Prudnikov plays for FC Amkar Perm and FC Tom Tomsk.\nSemantic graph: [[\"Aleksandr Prudnikov\", \"club\", \"FC Amkar Perm\"], [\"Aleksandr Prudnikov\", \"club\", \"FC Tom Tomsk\"]].\nText: " + text + "\nSemantic graph:"
    elif prompt_type == "three":
        prompt = "Transform the text into a semantic graph, which means, extract the triples from the text in format of lists like the following, [[\"subject\", \"predicate\", \"object\"], [\"subject\", \"predicate\", \"object\"], [\"subject\", \"predicate\", \"object\"]].\nExample 1: \nText: Aarhus Airport serves the city of Aarhus, Denmark.\nSemantic graph: [[\"Aarhus Airport\", \"city Served\", \"Aarhus Denmark\"]].\nExample 2 :\nText: Aleksandr Prudnikov plays for FC Amkar Perm and FC Tom Tomsk.\nSemantic graph: [[\"Aleksandr Prudnikov\", \"club\", \"FC Amkar Perm\"], [\"Aleksandr Prudnikov\", \"club\", \"FC Tom Tomsk\"]].\nExample 3:\nText: The completion date of Adare Manor is 1862 and was started in 1700 and designed by Augustus Pugin.\nSemantic graph: [[\"Adare Manor\", \"completion Date\", \"1862\"], [\"Adare Manor\", \"architect\", \"Augustus Pugin\"], [\"Adare Manor\", \"building Start Date\", \"1700\"]].\nText: " + text + "\nSemantic graph:"
    elif prompt_type == "four":
        prompt = "Transform the text into a semantic graph, which means, extract the triples from the text in format of lists like the following, [[\"subject\", \"predicate\", \"object\"], [\"subject\", \"predicate\", \"object\"], [\"subject\", \"predicate\", \"object\"]].\nExample 1: \nText: Aarhus Airport serves the city of Aarhus, Denmark.\nSemantic graph: [[\"Aarhus Airport\", \"city Served\", \"Aarhus Denmark\"]].\nExample 2 :\nText: Aleksandr Prudnikov plays for FC Amkar Perm and FC Tom Tomsk.\nSemantic graph: [[\"Aleksandr Prudnikov\", \"club\", \"FC Amkar Perm\"], [\"Aleksandr Prudnikov\", \"club\", \"FC Tom Tomsk\"]].\nExample 3:\nText: The completion date of Adare Manor is 1862 and was started in 1700 and designed by Augustus Pugin.\nSemantic graph: [[\"Adare Manor\", \"completion Date\", \"1862\"], [\"Adare Manor\", \"architect\", \"Augustus Pugin\"], [\"Adare Manor\", \"building Start Date\", \"1700\"]]. \nExample 4:\nText: The building at 320 South Boston Street, formerly called the Exchange National Bank Building, is 121.92 metres, has 22 floors, and was completed in 1929.\nSemantic graph: [[\"320 South Boston Building\", \"height\", \"121.92 metres\"], [\"320 South Boston Building\", \"former Name\", \"Exchange National Bank Building\"], [\"320 South Boston Building\", \"completion Date\", \"1929\"], [\"320 South Boston Building\", \"floor Count\", \"22\"]].\nText: " + text + "\nSemantic graph:"
    elif prompt_type == "five":
        prompt = "Transform the text into a semantic graph, which means, extract the triples from the text in format of lists like the following, [[\"subject\", \"predicate\", \"object\"], [\"subject\", \"predicate\", \"object\"], [\"subject\", \"predicate\", \"object\"]].\nExample 1: \nText: Aarhus Airport serves the city of Aarhus, Denmark.\nSemantic graph: [[\"Aarhus Airport\", \"city Served\", \"Aarhus Denmark\"]].\nExample 2 :\nText: Aleksandr Prudnikov plays for FC Amkar Perm and FC Tom Tomsk.\nSemantic graph: [[\"Aleksandr Prudnikov\", \"club\", \"FC Amkar Perm\"], [\"Aleksandr Prudnikov\", \"club\", \"FC Tom Tomsk\"]].\nExample 3:\nText: The completion date of Adare Manor is 1862 and was started in 1700 and designed by Augustus Pugin.\nSemantic graph: [[\"Adare Manor\", \"completion Date\", \"1862\"], [\"Adare Manor\", \"architect\", \"Augustus Pugin\"], [\"Adare Manor\", \"building Start Date\", \"1700\"]]. \nExample 4:\nText: The building at 320 South Boston Street, formerly called the Exchange National Bank Building, is 121.92 metres, has 22 floors, and was completed in 1929.\nSemantic graph: [[\"320 South Boston Building\", \"height\", \"121.92 metres\"], [\"320 South Boston Building\", \"former Name\", \"Exchange National Bank Building\"], [\"320 South Boston Building\", \"completion Date\", \"1929\"], [\"320 South Boston Building\", \"floor Count\", \"22\"]].\nExample 5:\nText: Ajoblanco or ajo blanco, a dish made of bread, almonds, garlic, water, olive oil, is from the Andalusia region of Spain.\nSemantic graph: [[\"Ajoblanco\", \"country\", \"Spain\"], [\"Ajoblanco\", \"main Ingredient\", \"Bread almonds garlic water olive oil\"], [\"Ajoblanco\", \"region\", \"Andalusia\"], [\"Ajoblanco\", \"alternative Name\", \"Ajo blanco\"], [\"Ajoblanco\", \"ingredient\", \"Garlic\"]].\nText: " + text + "\nSemantic graph:"
    elif prompt_type == "six":
        prompt = "Transform the text into a semantic graph, which means, extract the triples from the text in format of lists like the following, [[\"subject\", \"predicate\", \"object\"], [\"subject\", \"predicate\", \"object\"], [\"subject\", \"predicate\", \"object\"]].\nExample 1: \nText: Aarhus Airport serves the city of Aarhus, Denmark.\nSemantic graph: [[\"Aarhus Airport\", \"city Served\", \"Aarhus Denmark\"]].\nExample 2 :\nText: Aleksandr Prudnikov plays for FC Amkar Perm and FC Tom Tomsk.\nSemantic graph: [[\"Aleksandr Prudnikov\", \"club\", \"FC Amkar Perm\"], [\"Aleksandr Prudnikov\", \"club\", \"FC Tom Tomsk\"]].\nExample 3:\nText: The completion date of Adare Manor is 1862 and was started in 1700 and designed by Augustus Pugin.\nSemantic graph: [[\"Adare Manor\", \"completion Date\", \"1862\"], [\"Adare Manor\", \"architect\", \"Augustus Pugin\"], [\"Adare Manor\", \"building Start Date\", \"1700\"]]. \nExample 4:\nText: The building at 320 South Boston Street, formerly called the Exchange National Bank Building, is 121.92 metres, has 22 floors, and was completed in 1929.\nSemantic graph: [[\"320 South Boston Building\", \"height\", \"121.92 metres\"], [\"320 South Boston Building\", \"former Name\", \"Exchange National Bank Building\"], [\"320 South Boston Building\", \"completion Date\", \"1929\"], [\"320 South Boston Building\", \"floor Count\", \"22\"]].\nExample 5:\nText: Ajoblanco or ajo blanco, a dish made of bread, almonds, garlic, water, olive oil, is from the Andalusia region of Spain.\nSemantic graph: [[\"Ajoblanco\", \"country\", \"Spain\"], [\"Ajoblanco\", \"main Ingredient\", \"Bread almonds garlic water olive oil\"], [\"Ajoblanco\", \"region\", \"Andalusia\"], [\"Ajoblanco\", \"alternative Name\", \"Ajo blanco\"], [\"Ajoblanco\", \"ingredient\", \"Garlic\"]].\nExample 6:\nText: Acharya Institute of Technology is located at campus In Soldevanahalli, Acharya Dr. Sarvapalli Radhakrishnan Road, Hessarghatta Main Road, Bangalore – 560090 in Bangalore, India. Its director is Dr. G. P. Prabhukumar and its affiliation is Visvesvaraya Technological University. The Institute has 700 postgraduate students.\nSemantic graph: [[\"Acharya Institute of Technology\", \"city\", \"Bangalore\"], [\"Acharya Institute of Technology\", \"director\", \"Dr. G. P. Prabhukumar\"], [\"Acharya Institute of Technology\", \"country\", \"India\"], [\"Acharya Institute of Technology\", \"number Of Postgraduate Students\", \"700\"], [\"Acharya Institute of Technology\", \"campus\", \"In Soldevanahalli Acharya Dr. Sarvapalli Radhakrishnan Road Hessarghatta Main Road Bangalore - 560090.\"], [\"Acharya Institute of Technology\", \"affiliation\", \"Visvesvaraya Technological University\"]].\nText: " + text + "\nSemantic graph:"
    else: 
        prompt = "Transform the text into a semantic graph, which means, extract the triples from the text in format of lists like the following, [[\"subject\", \"predicate\", \"object\"], [\"subject\", \"predicate\", \"object\"], [\"subject\", \"predicate\", \"object\"]].\nExample 1: \nText: Aarhus Airport serves the city of Aarhus, Denmark.\nSemantic graph: [[\"Aarhus Airport\", \"city Served\", \"Aarhus Denmark\"]].\nExample 2 :\nText: Aleksandr Prudnikov plays for FC Amkar Perm and FC Tom Tomsk.\nSemantic graph: [[\"Aleksandr Prudnikov\", \"club\", \"FC Amkar Perm\"], [\"Aleksandr Prudnikov\", \"club\", \"FC Tom Tomsk\"]].\nExample 3:\nText: The completion date of Adare Manor is 1862 and was started in 1700 and designed by Augustus Pugin.\nSemantic graph: [[\"Adare Manor\", \"completion Date\", \"1862\"], [\"Adare Manor\", \"architect\", \"Augustus Pugin\"], [\"Adare Manor\", \"building Start Date\", \"1700\"]]. \nExample 4:\nText: The building at 320 South Boston Street, formerly called the Exchange National Bank Building, is 121.92 metres, has 22 floors, and was completed in 1929.\nSemantic graph: [[\"320 South Boston Building\", \"height\", \"121.92 metres\"], [\"320 South Boston Building\", \"former Name\", \"Exchange National Bank Building\"], [\"320 South Boston Building\", \"completion Date\", \"1929\"], [\"320 South Boston Building\", \"floor Count\", \"22\"]].\nExample 5:\nText: Ajoblanco or ajo blanco, a dish made of bread, almonds, garlic, water, olive oil, is from the Andalusia region of Spain.\nSemantic graph: [[\"Ajoblanco\", \"country\", \"Spain\"], [\"Ajoblanco\", \"main Ingredient\", \"Bread almonds garlic water olive oil\"], [\"Ajoblanco\", \"region\", \"Andalusia\"], [\"Ajoblanco\", \"alternative Name\", \"Ajo blanco\"], [\"Ajoblanco\", \"ingredient\", \"Garlic\"]].\nExample 6:\nText: Acharya Institute of Technology is located at campus In Soldevanahalli, Acharya Dr. Sarvapalli Radhakrishnan Road, Hessarghatta Main Road, Bangalore – 560090 in Bangalore, India. Its director is Dr. G. P. Prabhukumar and its affiliation is Visvesvaraya Technological University. The Institute has 700 postgraduate students.\nSemantic graph: [[\"Acharya Institute of Technology\", \"city\", \"Bangalore\"], [\"Acharya Institute of Technology\", \"director\", \"Dr. G. P. Prabhukumar\"], [\"Acharya Institute of Technology\", \"country\", \"India\"], [\"Acharya Institute of Technology\", \"number Of Postgraduate Students\", \"700\"], [\"Acharya Institute of Technology\", \"campus\", \"In Soldevanahalli Acharya Dr. Sarvapalli Radhakrishnan Road Hessarghatta Main Road Bangalore - 560090.\"], [\"Acharya Institute of Technology\", \"affiliation\", \"Visvesvaraya Technological University\"]].\nExample 7:\nText: Alan Shepard was born on Nov 18, 1923 in New Hampshire, US. He graduated from NWC with an M.A. in 1957. He passed away in California on July 21, 1998.\nSemantic graph: [[\"Alan Shepard\", \"status\", \"Deceased\"], [\"Alan Shepard\", \"alma Mater\", \"NWC M.A. 1957\"], [\"Alan Shepard\", \"death Place\", \"California\"], [\"Alan Shepard\", \"birth Place\", \"New Hampshire\"], [\"Alan Shepard\", \"death Date\", \"1998-07-21\"], [\"Alan Shepard\", \"nationality\", \"United States\"], [\"Alan Shepard\", \"birth Date\", \"1923-11-18\"]].\nText: " + text + "\nSemantic graph:"
    return prompt

def main(test_tgt, base_model_id):
    # Load text data
    #text = load_text_data('webnlg_data/test_data/test.target')
    text = load_text_data(test_tgt)

    # Define the base model ID for the LLM
    #base_model_id = "mistralai/Mistral-7B-v0.1" # "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf" or "berkeley-nest/Starling-LM-7B-alpha"

    # Configure quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Initialize base model and tokenizer
    base_model = initialize_base_model(base_model_id, bnb_config)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True)
    
    # Uncomment the following line if generating with fine-tuned models
    #ft_model = PeftModel.from_pretrained(base_model, "mistral-webNLG2020-finetune/checkpoint-500") # "llama2-7b-webNLG2020-finetune/checkpoint-500", "llama2-13b-webNLG2020-finetune/checkpoint-500" or "starling-webNLG2020-finetune/checkpoint-500"
    
    # List of prompt types
    prompt_types = ["zero", "one", "two", "three", "four", "five", "six", "seven"]

    # Loop through each prompt type
    for prompt_type in prompt_types:
        process_prompt(prompt_type, text, base_model, tokenizer) #  # Pass ft_model instead of base_model for generating with fine-tuned models

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file_target", default=None, type=str, required=True)
    
    parser.add_argument("--base_model_name", default="mistral")

    args = parser.parse_args()
    
     # Define the base model ID for the Language Model (LLM)    
    if args.base_model_name = "mistral":
        base_model_id = "mistralai/Mistral-7B-v0.1"
    elif args.base_model_name = "starling":
        base_model_id = "berkeley-nest/Starling-LM-7B-alpha"
    elif args.base_model_name = "llama_7b:"
        base_model_id = "meta-llama/Llama-2-7b-hf"
    else : 
        base_model_id = "meta-llama/Llama-2-13b-hf"
    
    
    main(args.pred_file, base_model_id)

