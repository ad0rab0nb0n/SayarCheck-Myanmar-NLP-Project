import re


def create_break_pattern():
    """
    Create the same Myanmar syllable-breaking pattern
    used by the dataset creator.
    """

    my_consonant = r"က-အ"
    other_char = r"ဣဤဥဦဧဩဪဿ၌၍၏၀-၉၊။"
    symbols = r"!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~"

    subscript_symbol = r"္"
    a_that = r"်"

    return re.compile(
        r"((?<!" + subscript_symbol + r")[" + my_consonant + r"]"
        r"(?![" + a_that + subscript_symbol + r"])"
        + r"|[" + other_char + r"]"
        + r"|[" + symbols + r"])"
    )


break_pattern = create_break_pattern()


def break_syllables_myanmar(text, separator="|X|"):
    """
    Break Myanmar text into syllable-like tokens.
    """

    segmented = break_pattern.sub(
        separator + r"\1",
        text
    )

    if segmented.startswith(separator):
        segmented = segmented[len(separator):]

    return [
        s
        for s in segmented.split(separator)
        if s
    ]


def is_myanmar(text):
    """
    Check whether text contains Myanmar characters.
    """

    return bool(
        re.search(r"[\u1000-\u109F]", text)
    )
def tokenize_chunk(chunk):
    """
    Tokenize Myanmar and non-Myanmar text.

    Myanmar text is broken into syllables.
    Non-Myanmar text is separated around punctuation
    while keeping normal English/number groups together.
    """

    if not chunk:
        return []

    tokens = []

    parts = re.split(
        r"([\u1000-\u109F]+)",
        chunk
    )

    for part in parts:

        if not part:
            continue

        # Myanmar section
        if re.search(r"[\u1000-\u109F]", part):

            syllables = break_syllables_myanmar(part)
            tokens.extend(syllables)

        # Non-Myanmar section
        else:

            non_myanmar_tokens = re.findall(
                r"[A-Za-z]+(?:[-'][A-Za-z]+)*"
                r"|\d+(?:\.\d+)*"
                r"|[^\w\s]"
                r"|_+",
                part
            )

            tokens.extend(
                token
                for token in non_myanmar_tokens
                if token
            )

    return tokens

def tokenize_text(text):
    """
    Tokenize a complete Myanmar text.
    """

    segments = text.split()

    all_tokens = []

    for segment in segments:

        tokens = tokenize_chunk(segment)

        all_tokens.extend(tokens)

    return all_tokens