document.addEventListener("DOMContentLoaded", () => {

    const youtubeUrlInput = document.getElementById("youtube-url");
    const questionInput = document.getElementById("question");
    const askButton = document.getElementById("ask-btn");
    const videoStatus = document.getElementById("video-status");
    const answerBox = document.getElementById("answer");

    // Check that all required elements exist
    if (!youtubeUrlInput) {
        console.error("youtube-url element not found");
        return;
    }

    if (!questionInput) {
        console.error("question element not found");
        return;
    }

    if (!askButton) {
        console.error("ask-btn element not found");
        return;
    }

    if (!answerBox) {
        console.error("answer element not found");
        return;
    }


    // ==========================================
    // ASK QUESTION
    // ==========================================

    askButton.addEventListener("click", async () => {

        const url = youtubeUrlInput.value.trim();
        const question = questionInput.value.trim();


        // --------------------------------------
        // Validate URL
        // --------------------------------------

        if (!url) {

            videoStatus.textContent =
                "Please paste a YouTube URL.";

            return;
        }


        // --------------------------------------
        // Validate question
        // --------------------------------------

        if (!question) {

            videoStatus.textContent =
                "Please enter a question.";

            return;
        }


        // --------------------------------------
        // Loading state
        // --------------------------------------

        askButton.disabled = true;
        askButton.textContent = "Processing...";

        videoStatus.textContent =
            "Reading the video and generating your answer...";

        answerBox.textContent =
            "Please wait...";


        try {

            // ----------------------------------
            // Send URL + question to FastAPI
            // ----------------------------------

            const response = await fetch("/ask", {

                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({
                    url: url,
                    question: question
                })
            });


            // ----------------------------------
            // Get backend response
            // ----------------------------------

            const data = await response.json();


            console.log("Backend response:", data);


            // ----------------------------------
            // Handle error
            // ----------------------------------

            if (!response.ok || data.error) {

                answerBox.textContent =
                    data.error || "Something went wrong.";

                videoStatus.textContent =
                    "Unable to process the video.";

                return;
            }


            // ----------------------------------
            // Display answer
            // ----------------------------------

            answerBox.textContent =
                data.answer;

            videoStatus.textContent =
                "Answer generated successfully.";

        }


        catch (error) {

            console.error("Request error:", error);

            answerBox.textContent =
                "Something went wrong. Please check the terminal.";

            videoStatus.textContent =
                "Unable to connect to the server.";
        }


        finally {

            askButton.disabled = false;
            askButton.textContent = "Ask Question";

        }

    });

});