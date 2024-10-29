document.getElementById("uploadForm").addEventListener("submit", async function (event) {
    event.preventDefault();

    const fileInput = document.getElementById("fileInput");
    const resultDiv = document.getElementById("result");

    if (fileInput.files.length === 0) {
        resultDiv.innerText = "Por favor, selecione uma imagem.";
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();
        if (response.ok) {
            resultDiv.innerText = `Resultado: ${data.prediction} (Confiança: ${(data.confidence * 100).toFixed(2)}%)`;
        } else {
            resultDiv.innerText = `Erro: ${data.error}`;
        }
    } catch (error) {
        resultDiv.innerText = `Erro na requisição: ${error.message}`;
    }
});
