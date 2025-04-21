import numpy as np
import spacy
import pickle
from collections import defaultdict
import os


# ==== DATA ==== 
context_text = """
Mickey Mouse is an animated cartoon character created by Walt Disney and Ub Iwerks. A cheerful and plucky anthropomorphic mouse, Mickey made his first public appearance in Steamboat Willie on November 18, 1928. Standing approximately 2'3" (68.6 cm) and weighing 23 pounds (10.4 kg), he is easily recognized by his round ears, red shorts with white buttons, white gloves, and yellow shoes. He speaks in a falsetto voice and often uses 1930s slang such as "swell" and "gee." Mickey is portrayed as an underdog who overcomes adversity through wit and determination.

Mickey gained popularity through his long-running short film series. His simple "rubber hose" animation style allows for expressive emotion and exaggerated physical comedy. His adaptability lets him take on various roles and settings while maintaining his core traits. He expanded into other media, beginning with a comic strip by Floyd Gottfredson in 1930. In 1940, he appeared in Fantasia, with a modern redesign by animator Fred Moore. In the 1950s, Mickey became a television icon with The Mickey Mouse Club and a symbol of the Disney theme parks.

Mickey quickly became a cultural icon and the most recognizable cartoon character worldwide. He was especially cherished by Walt Disney, who originally voiced the character. Disney often credited Mickey as the foundation of his success, famously stating, "It was all started by a mouse."

As the mascot of The Walt Disney Company, Mickey has appeared in over 130 films and received numerous accolades. In 1932, Walt Disney received an honorary Academy Award for Mickey's creation. In 1978, Mickey became the first cartoon character to receive a star on the Hollywood Walk of Fame. He remains the highest-grossing animated character of all time and the fourth highest-grossing media franchise overall.
"""

# ==== QUESTION-ANSWER DATA ==== 
qa_data = [
    {"question": "When did Mickey first appear?", "answer": "1928"},
    {"question": "Who created Mickey Mouse?", "answer": "Walt Disney and Ub Iwerks"},
    {"question": "What color are Mickey's shoes?", "answer": "Yellow"},
    {"question": "What is Mickey's height?", "answer": "2 feet 3 inches"},
    {"question": "What year did Mickey make his first public appearance?", "answer": "1928"},
    {"question": "Who voiced Mickey Mouse?", "answer": "Walt Disney"},
    {"question": "In which film did Mickey first appear?", "answer": "Steamboat Willie"},
    {"question": "What is Mickey's weight?", "answer": "23 pounds"},
    {"question": "What is the design style of Mickey Mouse?", "answer": "Rubberhose animation"},
    {"question": "What type of character is Mickey Mouse?", "answer": "A cheerful, anthropomorphic mouse"},
    {"question": "What is Mickey's personality like?", "answer": "An underdog who overcomes adversity with quick wit"},
    {"question": "When was Mickey Mouse first designed?", "answer": "1928"},
    {"question": "Who was responsible for Mickey's redesign in 1940?", "answer": "Fred Moore"},
    {"question": "What iconic 1930s slang does Mickey use?", "answer": "Swell and gee"},
    {"question": "What is Mickey Mouse's primary role in Disney?", "answer": "The mascot of The Walt Disney Company"},
    {"question": "Which film gave Mickey Mouse his modern redesign?", "answer": "Fantasia"},
    {"question": "What was Mickey Mouse's first comic strip?", "answer": "A comic strip by Floyd Gottfredson in 1930"},
    {"question": "What year did Mickey appear in the Mickey Mouse Club?", "answer": "1955"},
    {"question": "What significant role did Mickey play in Disney theme parks?", "answer": "He became an icon for the Disney theme parks"},
    {"question": "Which video game series prominently featured Mickey Mouse?", "answer": "Kingdom Hearts"},
    {"question": "What is Mickey's most iconic physical feature?", "answer": "His round ears"},
    {"question": "What is the height of Mickey Mouse in centimeters?", "answer": "68.58 centimeters"},
    {"question": "How much does Mickey Mouse weigh in kilograms?", "answer": "10.43 kilograms"},
    {"question": "What are Mickey's clothes like?", "answer": "Red shorts with white buttons, white gloves, and yellow shoes"},
    {"question": "What kind of voice does Mickey speak in?", "answer": "A falsetto voice"},
    {"question": "What kind of adversity does Mickey face?", "answer": "Larger-than-life adversity that he overcomes through wit"},
    {"question": "When did Mickey make his film debut?", "answer": "In 1928 with Steamboat Willie"},
    {"question": "Which year did Mickey appear in Fantasia?", "answer": "1940"},
    {"question": "What genre of animation does Mickey Mouse belong to?", "answer": "Cartoon animation"},
    {"question": "What feature made Mickey Mouse's animation style special?", "answer": "His simplistic 'rubberhose' design"},
    {"question": "How did Mickey become recognized as a cultural icon?", "answer": "Through his long-running short films and his unique personality"},
    {"question": "What kind of character is Mickey Mouse portrayed as?", "answer": "An underdog who overcomes adversity"},
    {"question": "What year did Mickey become the first cartoon character to receive a star on the Hollywood Walk of Fame?", "answer": "1978"},
    {"question": "What was Mickey Mouse's impact on the cartoon industry?", "answer": "He influenced following cartoons and became the most recognizable character"},
    {"question": "What did Walt Disney once say about Mickey Mouse?", "answer": "'It was all started by a mouse.'"},
    {"question": "What movie featured Mickey Mouse in a modern redesign?", "answer": "Fantasia"},
    {"question": "What kind of design did Mickey Mouse originally have?", "answer": "A simplistic rubberhose design"},
    {"question": "Who voiced Mickey Mouse for most of his early years?", "answer": "Walt Disney"},
    {"question": "How many films has Mickey appeared in?", "answer": "Over 130 films"},
    {"question": "What was Mickey's most important cultural role?", "answer": "He became a symbol of The Walt Disney Company"},
    {"question": "What year did Mickey Mouse receive an honorary Academy Award?", "answer": "1932"},
    {"question": "What is Mickey's significance in pop culture?", "answer": "He is one of the most recognizable and popular characters worldwide"},
    {"question": "What video game featured Mickey in 1990?", "answer": "Castle of Illusion"},
    {"question": "How much did Mickey contribute to the Disney company?", "answer": "He is considered the foundation of Disney's success"},
    {"question": "What did Mickey Mouse represent for Walt Disney?", "answer": "Walt Disney's alter-ego and the core of his success"},
    {"question": "What animated film did Mickey appear in 1940?", "answer": "Fantasia"},
    {"question": "How did Mickey impact the popularity of animation?", "answer": "He helped make animation a mainstream form of entertainment"},
    {"question": "What did Mickey Mouse symbolize for Walt Disney?", "answer": "The beginning of his career and the success of the Disney company"},
    {"question": "How is Mickey recognized in pop culture?", "answer": "As the most popular and recognizable cartoon character"},
    {"question": "In what year did Mickey become the highest-grossing animated character?", "answer": "Today, Mickey stands as the highest-grossing animated character of all time"},
    {"question": "Where did Mickey Mouse first appear in 1928?", "answer": "In Steamboat Willie"},
    {"question": "What is Mickey Mouse's famous quote?", "answer": "'It was all started by a mouse.'"},
    {"question": "When did Mickey appear on the Hollywood Walk of Fame?", "answer": "1978"},
    {"question": "What is Mickey Mouse's legacy in animation?", "answer": "He paved the way for modern animated characters and films"},
    {"question": "What kind of personality does Mickey have?", "answer": "Cheerful, plucky, and determined"},
    {"question": "How did Mickey's character evolve over time?", "answer": "He was redesigned in 1940 and became a cultural icon"},
    {"question": "What year was Mickey Mouse's comic strip published?", "answer": "1930"},
    {"question": "Where did Mickey become an icon for the Disney parks?", "answer": "At Disneyland"},
    {"question": "How was Mickey's voice portrayed?", "answer": "By Walt Disney himself"},
    {"question": "How did Mickey influence future cartoon characters?", "answer": "He became the blueprint for subsequent animated characters"},
    {"question": "What is Mickey's iconic look?", "answer": "Round ears, red shorts, white gloves, and yellow shoes"},
    {"question": "Who was the first person to receive a star on the Hollywood Walk of Fame?", "answer": "Mickey Mouse"},
    {"question": "How did Mickey's design influence animation?", "answer": "His simple and expressive design allowed for exaggerated physical comedy"},
    {"question": "In which film did Mickey make his feature-length debut?", "answer": "Fantasia"},
    {"question": "How did Mickey make an impact in video games?", "answer": "Through games like Castle of Illusion and Kingdom Hearts"},
    {"question": "What year did Mickey receive numerous accolades?", "answer": "Over the years, Mickey has received many awards and recognitions"},
    {"question": "How many films did Mickey appear in to become iconic?", "answer": "In over 130 films"},
    {"question": "What was the significance of Mickey's design in animation?", "answer": "His rubberhose animation style made him both relatable and comically exaggerated"},
    {"question": "What makes Mickey Mouse a cultural icon?", "answer": "His widespread popularity and representation of Disney's values"},
    {"question": "What made Mickey a success in both film and television?", "answer": "His adaptability across different media, including film and TV shows"},
    {"question": "What animated film featured Mickey Mouse in 1940?", "answer": "Fantasia"},
    {"question": "What was Mickey's importance in the 1950s?", "answer": "He became a TV star with The Mickey Mouse Club"},
    {"question": "What year did Disneyland open with Mickey Mouse as an icon?", "answer": "1955"},
    {"question": "Who were Mickey Mouse's creators?", "answer": "Walt Disney and Ub Iwerks"},
    {"question": "What character trait defines Mickey?", "answer": "He is an underdog who uses wit to overcome adversity"},
    {"question": "What major event happened for Mickey Mouse in 1932?", "answer": "Walt Disney received an honorary Academy Award for creating Mickey Mouse"},
    {"question": "How did Mickey's character evolve into a global icon?", "answer": "Through his consistent presence in films, comics, and merchandise"}
]



nlp = spacy.load("en_core_web_sm")

# ==== CONFIGURATION ====
embedding_dim = 64
np.random.seed(42)
token_vectors = {}
idf_scores = defaultdict(lambda: 1.0)

# ==== TOKENIZER ====
def tokenize(text):
    return text.lower().replace('.', '').replace(',', '').replace('"', '').replace("'", '').split()

# ==== TOKEN VECTORS ====
def get_token_vector(token):
    if token not in token_vectors:
        token_vectors[token] = np.random.uniform(-1, 1, embedding_dim)
    return token_vectors[token]

# ==== IDF ====
def compute_idf(corpus):
    doc_count = defaultdict(int)
    total_docs = len(corpus)
    for doc in corpus:
        seen = set(tokenize(doc))
        for token in seen:
            doc_count[token] += 1
    for token, count in doc_count.items():
        idf_scores[token] = np.log((total_docs + 1) / (count + 1)) + 1

# ==== EMBEDDER: TF-IDF + POSITIONAL WEIGHTING ====
def embed(text):
    tokens = tokenize(text)
    if not tokens:
        return np.zeros(embedding_dim)
    weighted = []
    for i, token in enumerate(tokens):
        tf = 1
        idf = idf_scores.get(token, 1.0)
        pos_weight = 1 / (i + 1)
        vec = get_token_vector(token) * idf * pos_weight
        weighted.append(vec)
    return np.mean(weighted, axis=0)

# ==== SIMPLE ATTENTION OVER CONTEXT ====
def attention_embed(context, question):
    context_tokens = tokenize(context)
    question_vec = embed(question)
    token_vecs = [get_token_vector(t) for t in context_tokens]
    scores = [np.dot(v, question_vec) / (np.linalg.norm(v) * np.linalg.norm(question_vec) + 1e-8) for v in token_vecs]
    weights = np.exp(scores) / np.sum(np.exp(scores))
    attended = np.sum([w * v for w, v in zip(weights, token_vecs)], axis=0)
    return attended

# ==== SAVE AND LOAD FUNCTIONS ====
def save_model(filename="model.pkl"):
    with open(filename, "wb") as f:
        model_data = {
            "token_vectors": token_vectors,
            "idf_scores": dict(idf_scores)
        }
        pickle.dump(model_data, f)
    print(f"Model saved to {filename} successfully.")

def load_model(filename="model.pkl"):
    global token_vectors, idf_scores
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        try:
            with open(filename, "rb") as f:
                model_data = pickle.load(f)
                token_vectors = model_data.get("token_vectors", {})
                idf_scores = model_data.get("idf_scores", {})
            print("Model loaded from", filename)
        except Exception as e:
            print(f"Error loading model: {e}")
            token_vectors = {}
            idf_scores = {}
    else:
        print(f"No model file found or file is empty, starting from scratch.")
        token_vectors = {}
        idf_scores = {}

# ==== SAMPLE DATA ====
qa_data.extend([
    {
        "question": "Why did Walt Disney consider Mickey his alter-ego?",
        "answer": "Because Walt Disney voiced Mickey and viewed him as a representation of himself."
    },
    {
        "question": "What qualities allowed Mickey to succeed despite being small?",
        "answer": "His quick wit and can-do spirit."
    },
    {
        "question": "How did Mickey contribute to Walt Disney’s career?",
        "answer": "Disney credited Mickey for his success, saying 'it was all started by a mouse'."
    },
    {
        "question": "What made Mickey adaptable to different stories and eras?",
        "answer": "His simple design and versatile character traits."
    }
])



# ==== LOAD MODEL ====
load_model()

# ==== IDF TRAINING ====
compute_idf([context_text] + [qa["question"] for qa in qa_data])

# ==== MODEL SETUP ====
input_dim = 64 * 2
hidden_dim = 128
output_dim = 64
lr = 0.01

W1 = np.random.randn(input_dim, hidden_dim) * 0.1
b1 = np.zeros(hidden_dim)
W2 = np.random.randn(hidden_dim, output_dim) * 0.1
b2 = np.zeros(output_dim)

def cosine_similarity(a, b):
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b) + 1e-8)
    return np.dot(a_norm, b_norm)

def forward(input_vec):
    h = np.maximum(0, np.dot(input_vec, W1) + b1)
    out = np.dot(h, W2) + b2
    return out, h

# ==== TRAINING LOOP ====
iteration = 0
mastered_questions = set()
total_score = 0
total_questions = 0

inferential_questions_added = False

while iteration < 1000:
    total_score = 0.0
    for sample in qa_data:
        q_vec = embed(sample["question"])
        ctx_vec = attention_embed(context_text, sample["question"])

        input_vec = np.concatenate([ctx_vec, q_vec])
        pred, h = forward(input_vec)

        pred_answer_vec = embed(sample["answer"])
        score = cosine_similarity(pred, pred_answer_vec)
        total_score += score

        # Mastery check
        if score >= 0.99:
            mastered_questions.add(sample["question"])

        # Backpropagation
        if score < 0.99:
            error = pred - pred_answer_vec
            dW2 = np.outer(h, error)
            db2 = error
            dh = np.dot(W2, error)
            dh[h <= 0] = 0
            dW1 = np.outer(input_vec, dh)
            db1 = dh
            W2 -= lr * dW2
            b2 -= lr * db2
            W1 -= lr * dW1
            b1 -= lr * db1

    avg_score = total_score / len(qa_data)

    if iteration % 10 == 0:
        print(f"Iteration {iteration}: Avg Score = {avg_score:.4f}, Mastered = {len(mastered_questions)}/{len(qa_data)}")

    # Stop early if average score is high enough
    if avg_score >= 0.991:
        print(f"Early stopping at iteration {iteration}, Avg Score = {avg_score:.4f}")
        
        # Add inferential questions and continue
        if not inferential_questions_added:
            inferential_questions_added = True
            new_qas = [
                {
                    "question": "Why did Walt Disney consider Mickey his alter-ego?",
                    "answer": "Because Walt Disney voiced Mickey and viewed him as a representation of himself."
                },
                {
                    "question": "How did Mickey contribute to Walt Disney’s career?",
                    "answer": "Disney credited Mickey for his success, saying 'it was all started by a mouse'."
                },
                {
                    "question": "What qualities helped Mickey overcome adversity?",
                    "answer": "His quick wit and can-do spirit."
                },
                {
                    "question": "What makes Mickey a timeless character?",
                    "answer": "His simple design and versatility across settings and eras."
                }
            ]
            qa_data.extend(new_qas)
            compute_idf([context_text] + [qa["question"] for qa in qa_data])  # Update IDF with new Qs
            print("Added inferential questions. Continuing training...")
        else:
            break  # Already added inferential questions and finished training on them

    iteration += 1


# ==== SAVE MODEL ====
save_model()