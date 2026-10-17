/* ============================================================
   PROCESS TEXT
   ============================================================ */

async function processText() {

    const input =
        document.getElementById("inputText");

    const output =
        document.getElementById("outputText");

    const status =
        document.getElementById("status");

    const text =
        input.value.trim();


    /* ========================================================
       EMPTY INPUT
       ======================================================== */

    if (!text) {

        output.innerHTML = `
            <div class="output-placeholder">

                <div class="placeholder-icon">
                    ✓
                </div>

                <p>
                    Please enter Myanmar text first.
                </p>

            </div>
        `;

        status.textContent = "";

        return;
    }


    /* ========================================================
       PROCESSING
       ======================================================== */

    output.innerHTML = `
        <div class="output-placeholder">

            <div class="placeholder-icon">
                ...
            </div>

            <p>
                Processing your text...
            </p>

        </div>
    `;


    status.textContent =
        "Running SayarCheck...";


    try {

        /* ====================================================
           SEND TEXT TO FLASK
           ==================================================== */

        const response =
            await fetch(
                "/process",
                {

                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({
                        text: text
                    })

                }
            );


        const data =
            await response.json();


        /* ====================================================
           ERROR HANDLING
           ==================================================== */

        if (!response.ok) {

            throw new Error(
                data.error ||
                "Processing failed"
            );

        }


        /* ====================================================
           PROCESSED OUTPUT
           ====================================================

           data.result
           = segmented sentence

           data.corrected_result
           = spelling-corrected sentence

           We display corrected_result.
        */

        output.innerHTML = "";


        const correctedResult =
            document.createElement("div");


        correctedResult.className =
            "segmented-result";


        correctedResult.textContent =
            data.corrected_result ||
            data.result;


        output.appendChild(
            correctedResult
        );


        /* ====================================================
           SPELLING CHECK SECTION
           ==================================================== */

        const spellingSection =
            document.createElement("div");


        spellingSection.className =
            "spelling-section";


        const title =
            document.createElement("h3");


        title.textContent =
            "Spelling Check";


        spellingSection.appendChild(
            title
        );


        let errorCount = 0;


        /* ====================================================
           DISPLAY SPELLING ERRORS
           ==================================================== */

        data.spelling.forEach(
            item => {

                if (!item.correct) {

                    errorCount++;


                    const errorBox =
                        document.createElement(
                            "div"
                        );


                    errorBox.className =
                        "spelling-error";


                    /* Misspelled word */

                    const word =
                        document.createElement(
                            "strong"
                        );


                    word.textContent =
                        item.word;


                    errorBox.appendChild(
                        word
                    );


                    /* Suggestions */

                    if (
                        item.suggestions &&
                        item.suggestions.length > 0
                    ) {

                        const suggestionText =
                            document.createElement(
                                "div"
                            );


                        suggestionText.className =
                            "suggestions";


                        suggestionText.textContent =
                            "Suggestions: " +
                            item.suggestions
                                .map(
                                    suggestion =>
                                        suggestion.word
                                )
                                .join(", ");


                        errorBox.appendChild(
                            suggestionText
                        );

                    }


                    spellingSection.appendChild(
                        errorBox
                    );

                }

            }
        );


        /* ====================================================
           NO SPELLING ERRORS
           ==================================================== */

        if (errorCount === 0) {

            const success =
                document.createElement(
                    "div"
                );


            success.className =
                "spelling-success";


            success.textContent =
                "✓ No spelling errors detected";


            spellingSection.appendChild(
                success
            );

        }


        /* Add spelling section */

        output.appendChild(
            spellingSection
        );


        /* ====================================================
           COMPLETE
           ==================================================== */

        status.textContent =
            "✓ SayarCheck completed successfully";


    } catch (error) {

        output.innerHTML = `
            <div class="output-placeholder">

                <div class="placeholder-icon">
                    !
                </div>

                <p>
                    Something went wrong.
                </p>

            </div>
        `;


        status.textContent =
            "Error: " +
            error.message;

    }

}


/* ============================================================
   USE EXAMPLE
   ============================================================ */

function useExample(text) {

    const input =
        document.getElementById(
            "inputText"
        );


    input.value = text;


    input.focus();


    input.selectionStart =
        input.value.length;


    input.selectionEnd =
        input.value.length;

}


/* ============================================================
   CLEAR TEXT
   ============================================================ */

function clearText() {

    document.getElementById(
        "inputText"
    ).value = "";


    document.getElementById(
        "outputText"
    ).innerHTML = `

        <div class="output-placeholder">

            <div class="placeholder-icon">
                ✓
            </div>

            <p>
                Your processed text
                will appear here.
            </p>

        </div>

    `;


    document.getElementById(
        "status"
    ).textContent = "";

}