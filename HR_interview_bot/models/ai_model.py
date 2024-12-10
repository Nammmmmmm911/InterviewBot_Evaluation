
from transformers import AutoTokenizer, AutoModelForCausalLM

def evaluate_answers_with_ai(answers, questions):
    try:
        # Ensure answers is a list
        if isinstance(answers, dict):
            answers = list(answers.values())  # Convert dictionary to list

        # Load the GPT-Neo model and tokenizer
        model_name = "EleutherAI/gpt-neo-1.3B"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        # Construct the prompt
        prompt = "Evaluate the following interview answers and provide constructive, specific feedback based on the content of the answers. Do not repeat feedback for different answers:\n\n"

        # Add questions and answers
        for idx, (question, answer) in enumerate(zip(questions, answers), start=1):
            prompt += f"Question {idx}: {question}\nAnswer: {answer}\n\n"
        
        if not prompt.strip():
            return {"error": "No answers provided to evaluate."}
        
        print("Constructed Prompt:", prompt)
        
        # Tokenize the input prompt
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
        
        # Generate the evaluation
        outputs = model.generate(**inputs, max_length=1000, num_return_sequences=1)

        # Decode the output
        evaluation_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        print("Evaluation Output:", evaluation_text)

        # Split the evaluation into lines
        evaluation_lines = evaluation_text.split("\n")

        # Process the response
        evaluation_results = {}
        answer_idx = 1

        for line in evaluation_lines:
            if line.startswith("Answer"):
                # Clean and store feedback for each answer
                feedback = line.strip()
                evaluation_results[answer_idx] = feedback
                answer_idx += 1

        # Handle any missing feedback
        for idx in range(1, len(answers) + 1):
            if idx not in evaluation_results:
                evaluation_results[idx] = "No feedback provided."

        return evaluation_results

    except Exception as e:
        return {"error": f"Error evaluating answers: {str(e)}"}
