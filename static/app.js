const fileInput = document.getElementById("documentFile");
const dropzone = document.getElementById("dropzone");
const fileName = document.getElementById("fileName");
const uploadButton = document.getElementById("uploadButton");
const askButton = document.getElementById("askButton");
const questionInput = document.getElementById("question");
const uploadFeedback = document.getElementById("uploadFeedback");
const askFeedback = document.getElementById("askFeedback");
const answerCard = document.getElementById("answerCard");
const sourcesWrapper = document.getElementById("sourcesWrapper");
const sourcesList = document.getElementById("sourcesList");
const sourceCount = document.getElementById("sourceCount");
const chunkCount = document.getElementById("chunkCount");
const healthDot = document.getElementById("healthDot");
const healthText = document.getElementById("healthText");
const documentsList = document.getElementById("documentsList");
const documentCount = document.getElementById("documentCount");

fileInput.addEventListener("change", () => {
    updateSelectedFile(fileInput.files[0]);
});

["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropzone.classList.add("dragging");
    });
});

["dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropzone.classList.remove("dragging");
    });
});

dropzone.addEventListener("drop", (event) => {
    const droppedFile = event.dataTransfer.files[0];
    if (!droppedFile) return;
    fileInput.files = event.dataTransfer.files;
    updateSelectedFile(droppedFile);
});

uploadButton.addEventListener("click", uploadDocument);
askButton.addEventListener("click", askQuestion);

questionInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
        event.preventDefault();
        askQuestion();
    }
});

window.addEventListener("load", () => {
    refreshHealth();
    refreshDocuments();
});

function updateSelectedFile(file) {
    if (!file) {
        fileName.textContent = "Dosyanı buraya bırak";
        return;
    }
    fileName.textContent = file.name;
}

async function uploadDocument() {
    const file = fileInput.files[0];
    if (!file) {
        showFeedback(uploadFeedback, "Önce bir doküman seçmelisin.", true);
        return;
    }

    setButtonLoading(uploadButton, true, "Indexleniyor...");
    clearFeedback(uploadFeedback);

    try {
        const formData = new FormData();
        formData.append("file", file);
        const response = await fetch("/documents", { method: "POST", body: formData });
        const data = await parseResponse(response);

        if (!response.ok) {
            throw new Error(data.detail || "Doküman indexlenemedi.");
        }

        chunkCount.textContent = data.chunks_indexed;
        showFeedback(
            uploadFeedback,
            `${data.filename} başarıyla indexlendi. ${data.chunks_indexed} chunk eklendi.`
        );
        await Promise.all([refreshHealth(), refreshDocuments()]);
    } catch (error) {
        showFeedback(uploadFeedback, error.message, true);
    } finally {
        setButtonLoading(uploadButton, false, "Dokümanı indexle");
    }
}

async function refreshDocuments() {
    try {
        const response = await fetch("/documents");
        const data = await parseResponse(response);
        if (!response.ok) throw new Error();
        renderDocuments(data.documents || []);
    } catch {
        documentCount.textContent = "-";
    }
}

function renderDocuments(documents) {
    documentsList.replaceChildren();
    documentCount.textContent = documents.length;

    if (!documents.length) {
        documentsList.appendChild(textElement("li", "documents-empty", "Henüz doküman yüklenmedi."));
        return;
    }

    documents.forEach((doc) => {
        const item = document.createElement("li");
        item.className = "document-item";

        const info = document.createElement("div");
        info.className = "document-info";
        const details = [`${doc.chunks} chunk`];
        if (doc.pages) details.push(`${doc.pages} sayfa`);
        info.append(
            textElement("strong", "document-name", doc.filename),
            textElement("span", "document-meta", details.join(" · "))
        );
        info.firstChild.title = doc.filename;

        const deleteButton = textElement("button", "document-delete", "×");
        deleteButton.type = "button";
        deleteButton.title = "Dokümanı sil";
        deleteButton.setAttribute("aria-label", `${doc.filename} dokümanını sil`);
        deleteButton.addEventListener("click", () => deleteDocument(doc, deleteButton));

        item.append(textElement("span", "file-symbol", "▤"), info, deleteButton);
        documentsList.appendChild(item);
    });
}

async function deleteDocument(doc, button) {
    if (!window.confirm(`"${doc.filename}" indexten silinsin mi? Bu dokümandan gelen kanıtlar artık kullanılmayacak.`)) {
        return;
    }

    button.disabled = true;
    clearFeedback(uploadFeedback);
    try {
        const response = await fetch(`/documents/${encodeURIComponent(doc.document_id)}`, { method: "DELETE" });
        const data = await parseResponse(response);
        if (!response.ok) {
            throw new Error(data.detail || "Doküman silinemedi.");
        }
        showFeedback(uploadFeedback, `${doc.filename} silindi.`);
    } catch (error) {
        showFeedback(uploadFeedback, error.message, true);
        button.disabled = false;
    }
    await Promise.all([refreshHealth(), refreshDocuments()]);
}

async function askQuestion() {
    const question = questionInput.value.trim();
    if (!question) {
        showFeedback(askFeedback, "Lütfen bir soru yazmalısın.", true);
        return;
    }

    setButtonLoading(askButton, true, "Cevap hazırlanıyor...");
    clearFeedback(askFeedback);

    try {
        const response = await fetch("/ask", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question }),
        });
        const data = await parseResponse(response);

        if (!response.ok) {
            throw new Error(data.detail || "Cevap oluşturulamadı.");
        }

        renderAnswer(data.answer, data.sources || [], data.verification);
    } catch (error) {
        showFeedback(askFeedback, error.message, true);
    } finally {
        setButtonLoading(askButton, false, "Cevap oluştur");
    }
}

async function refreshHealth() {
    try {
        const response = await fetch("/health");
        const data = await response.json();
        if (!response.ok) throw new Error();
        healthDot.classList.add("online");
        healthText.textContent = `Backend bağlı · ${data.indexed_chunks} chunk hazır`;
        chunkCount.textContent = data.indexed_chunks;
    } catch {
        healthDot.classList.remove("online");
        healthText.textContent = "Backend bağlantısı yok";
    }
}

async function parseResponse(response) {
    const data = await response.json().catch(() => ({}));
    if (Array.isArray(data.detail)) {
        data.detail = data.detail.map((item) => item.msg || "Geçersiz istek").join("; ");
    } else if (data.detail != null && typeof data.detail !== "string") {
        data.detail = "İstek işlenemedi.";
    }
    return data;
}

function renderAnswer(answer, sources, verification) {
    answerCard.className = "answer-card answer-ready";
    answerCard.replaceChildren();

    const icon = document.createElement("div");
    icon.className = "empty-icon";
    icon.textContent = "✦";

    const content = document.createElement("div");
    const label = document.createElement("p");
    label.className = "answer-label";
    label.textContent = "Evidence-based answer";
    const answerText = document.createElement("div");
    answerText.className = "answer-text";
    answerText.textContent = answer;
    content.append(label, answerText);
    const warnings = verificationWarnings(verification);
    if (warnings.length) {
        const warningBox = document.createElement("div");
        warningBox.className = "answer-warning";
        warningBox.setAttribute("role", "note");
        warnings.forEach((warning) => warningBox.appendChild(textElement("p", "", warning)));
        content.appendChild(warningBox);
    }
    answerCard.append(icon, content);

    renderSources(sources);
}

function verificationWarnings(verification) {
    if (!verification) return [];
    const warnings = [];
    const unsupported = verification.unsupported_numbers || [];
    const questionOnly = verification.question_only_numbers || [];
    const invalidCitations = verification.invalid_citations || [];
    if (unsupported.length) {
        warnings.push(
            `Cevaptaki şu sayılar kaynaklarda veya soruda bulunamadı: ${unsupported.join(", ")}. Bu değerleri kaynaklardan kontrol et.`
        );
    }
    if (questionOnly.length) {
        warnings.push(`Cevaptaki şu sayılar yalnız soruda geçiyor; kaynakta bulunamadı: ${questionOnly.join(", ")}.`);
    }
    if (invalidCitations.length) {
        const cited = invalidCitations.map((number) => `Kanıt ${number}`).join(", ");
        warnings.push(`Cevap var olmayan kaynaklara atıf yapıyor: ${cited}.`);
    }
    return warnings;
}

function renderSources(sources) {
    sourcesList.replaceChildren();

    if (!sources.length) {
        sourceCount.textContent = "Kaynak bulunamadı";
        sourcesWrapper.hidden = false;
        const empty = document.createElement("p");
        empty.className = "panel-description";
        empty.textContent = "Bu soru için yeterli relevance skoruna sahip kanıt bulunamadı.";
        sourcesList.appendChild(empty);
        return;
    }

    sourceCount.textContent = `${sources.length} kanıt parçası`;
    sourcesWrapper.hidden = false;

    sources.forEach((source, index) => {
        const card = document.createElement("article");
        card.className = "source-card";

        const top = document.createElement("div");
        top.className = "source-card-top";
        const badges = document.createElement("span");
        badges.className = "source-badges";
        if (source.cited) {
            const citedBadge = textElement("span", "source-cited", "Atıf");
            citedBadge.title = "Cevapta bu kaynak numarasına atıf yapıldı; iddianın doğruluğu denetlenmedi.";
            badges.appendChild(citedBadge);
        }
        badges.appendChild(textElement("span", "source-score", `Benzerlik ${source.score.toFixed(3)}`));
        top.append(
            textElement("span", "source-number", `KANIT ${String(source.evidence_number || index + 1).padStart(2, "0")}`),
            badges
        );

        const fileRow = document.createElement("div");
        fileRow.className = "source-file-row";
        fileRow.append(
            textElement("span", "file-symbol", "▤"),
            textElement("strong", "source-filename", source.filename || "Bilinmeyen dosya")
        );

        const meta = document.createElement("div");
        meta.className = "source-meta";
        meta.append(
            textElement("span", "", "Sayfa "),
            textElement("strong", "", source.page ?? "bilinmiyor")
        );

        const evidenceText = textElement("blockquote", "source-evidence", source.text || "Bu kaynak için metin sağlanmadı.");

        const details = document.createElement("details");
        details.className = "source-details";
        details.appendChild(textElement("summary", "", "Teknik kaynak bilgisi"));
        const detailText = `chunk_id: ${source.chunk_id || "-"}\ndocument_id: ${source.document_id || "-"}\nsource: ${source.source || "-"}`;
        const detailParagraph = textElement("p", "", detailText);
        detailParagraph.style.whiteSpace = "pre-wrap";
        details.appendChild(detailParagraph);

        card.append(top, fileRow, meta, evidenceText, details);
        sourcesList.appendChild(card);
    });
}

function textElement(tag, className, text) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    element.textContent = text;
    return element;
}

function setButtonLoading(button, loading, label) {
    button.disabled = loading;
    button.querySelector("span").textContent = label;
}

function showFeedback(element, message, error = false) {
    element.textContent = message;
    element.classList.toggle("error", error);
}

function clearFeedback(element) {
    element.textContent = "";
    element.classList.remove("error");
}
