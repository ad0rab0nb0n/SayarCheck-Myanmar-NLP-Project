from flask import Flask, render_template, request, jsonify
import torch
import json

from models.bilstm_crf import BiLSTMCRF
from preprocessing.unicode_normalizer import normalize_token
from preprocessing.syllable_tokenizer import tokenize_text
from spelling_checker import check_spelling, correct_text


# ============================================================
# FLASK APP
# ============================================================

app = Flask(
    __name__,
    static_folder="frontend",
    static_url_path="/static",
    template_folder="frontend"
)


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    device = torch.device("mps")

elif torch.cuda.is_available():
    device = torch.device("cuda")

else:
    device = torch.device("cpu")


print("Using device:", device)


# ============================================================
# LOAD VOCABULARY
# ============================================================

with open(
    "preprocessing/vocab.json",
    "r",
    encoding="utf-8"
) as f:

    word_to_id = json.load(f)


# ============================================================
# LOAD BILSTM-CRF MODEL
# ============================================================

print("Loading BiLSTM-CRF model...")


model = BiLSTMCRF(
    vocab_size=len(word_to_id),
    embedding_dim=128,
    hidden_dim=128,
    num_layers=2,
    num_tags=2,
    pad_idx=0
)


checkpoint = torch.load(
    "checkpoints/best_model.pt",
    map_location=device
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)


model = model.to(device)

model.eval()


print("Model loaded successfully!")


# ============================================================
# WORD / PHRASE SEGMENTATION
# ============================================================

def process_text(text):

    # --------------------------------------------------------
    # Step 1: Unicode normalization
    # --------------------------------------------------------

    text = normalize_token(text)


    # --------------------------------------------------------
    # Step 2: Tokenization
    # --------------------------------------------------------

    tokens = tokenize_text(text)


    if not tokens:
        return ""


    # --------------------------------------------------------
    # Step 3: Convert tokens to IDs
    # --------------------------------------------------------

    input_ids = []


    for token in tokens:

        token_id = word_to_id.get(
            token,
            word_to_id["<UNK>"]
        )

        input_ids.append(token_id)


    input_ids = torch.tensor(
        [input_ids],
        dtype=torch.long,
        device=device
    )


    # --------------------------------------------------------
    # Step 4: Attention mask
    # --------------------------------------------------------

    attention_mask = (
        input_ids != word_to_id["<PAD>"]
    )


    # --------------------------------------------------------
    # Step 5: BiLSTM-CRF prediction
    # --------------------------------------------------------

    with torch.no_grad():

        predictions = model(
            input_ids,
            attention_mask
        )


    tags = predictions[0]

    print("\nTOKENS:")
    print(tokens)

    print("\nPREDICTIONS:")
    print(predictions)

    print("\nTAGS TYPE:")
    print(type(tags))

    print("\nTAGS:")
    print(tags)

    # --------------------------------------------------------
    # Step 6: Create segmented sentence
    # --------------------------------------------------------

    result = ""


    for token, tag in zip(
        tokens,
        tags
    ):

        # B = 0
        # I = 1

        if tag == 0 and result:
            result += " "


        result += token


    return result


# ============================================================
# SPELLING CHECK
# ============================================================

def spelling_check(text):

    from spelling_checker import CONTEXT_RULES

    words = text.split()

    spelling_results = []

    # ========================================================
    # CHECK CONTEXT RULES FIRST
    # ========================================================

    context_errors = []

    for wrong_phrase, correct_phrase in CONTEXT_RULES.items():

        if wrong_phrase in text:

            context_errors.append({
                "word": wrong_phrase,
                "correct": False,
                "suggestions": [
                    {
                        "word": correct_phrase,
                        "distance": 0,
                        "frequency": 0
                    }
                ]
            })

    # ========================================================
    # CHECK INDIVIDUAL WORDS WITH SYMSPELL
    # ========================================================

    for word in words:

        suggestions = check_spelling(
            word,
            max_suggestions=5
        )

        # No suggestions
        if not suggestions:

            spelling_results.append({
                "word": word,
                "correct": True,
                "suggestions": []
            })

            continue

        best = suggestions[0]

        # Word is already correct
        if (
            best["word"] == word
            and best["distance"] == 0
        ):

            spelling_results.append({
                "word": word,
                "correct": True,
                "suggestions": []
            })

        else:

            spelling_results.append({
                "word": word,
                "correct": False,
                "suggestions": suggestions
            })

    # ========================================================
    # ADD CONTEXT-BASED ERRORS
    # ========================================================

    spelling_results = context_errors + spelling_results

    return spelling_results
# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# PROCESS TEXT
# ============================================================

@app.route(
    "/process",
    methods=["POST"]
)
def process():

    data = request.get_json()


    text = data.get(
        "text",
        ""
    ).strip()


    # --------------------------------------------------------
    # Empty input
    # --------------------------------------------------------

    if not text:

        return jsonify({

            "error":
                "No text provided."

        }), 400


    try:

        # ====================================================
        # STEP 1
        # WORD / PHRASE SEGMENTATION
        # ====================================================

        segmented_text = process_text(
            text
        )


        # ====================================================
        # STEP 2
        # SPELLING CHECK
        # ====================================================

        spelling_results = spelling_check(
            segmented_text
        )


        # ====================================================
        # STEP 3
        # AUTOMATIC SPELLING CORRECTION
        # ====================================================

        print("\nSEGMENTED TEXT:")
        print(segmented_text)

        corrected_text = correct_text(
            segmented_text
        )


        # ====================================================
        # SEND RESULT TO FRONTEND
        # ====================================================

        return jsonify({

            # Original segmented sentence
            "result":
                segmented_text,

            # Automatically corrected sentence
            "corrected_result":
                corrected_text,

            # Spelling suggestions
            "spelling":
                spelling_results

        })


    except Exception as e:

        return jsonify({

            "error":
                str(e)

        }), 500


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    print(
        "\n================================"
    )

    print(
        "SayarCheck"
    )

    print(
        "Myanmar Phrase Segmentation & Spelling Checker"
    )

    print(
        "================================"
    )

    print(
        "Open: http://127.0.0.1:5000"
    )

    print(
        "================================\n"
    )


    app.run(
        debug=True,
        port=5000
    )