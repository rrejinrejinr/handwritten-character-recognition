const { createWorker } = require('tesseract.js');
const path = require('path');
const fs = require('fs');

async function performOCR() {
    const args = process.argv.slice(2);
    if (args.length < 2) {
        console.error(JSON.stringify({ error: "Missing arguments. Usage: node ocr_bridge.js <image_path> <mode>" }));
        process.exit(1);
    }

    const imagePath = args[0];
    const mode = args[1];

    if (!fs.existsSync(imagePath)) {
        console.error(JSON.stringify({ error: `Image file does not exist: ${imagePath}` }));
        process.exit(1);
    }

    // Set whitelists based on mode
    let whitelist = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz';
    if (mode === 'digits') {
        whitelist = '0123456789';
    } else if (mode === 'characters') {
        whitelist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz';
    }

    try {
        const worker = await createWorker('eng');
        await worker.setParameters({
            tessedit_char_whitelist: whitelist,
            // Single character segmentation mode (PSM 10)
            tessedit_pageseg_mode: '10'
        });

        const { data: { text, confidence } } = await worker.recognize(imagePath);
        await worker.terminate();

        const cleanedText = text.trim();

        console.log(JSON.stringify({
            text: cleanedText || "?",
            confidence: confidence || 0.0
        }));
        process.exit(0);
    } catch (err) {
        console.error(JSON.stringify({ error: err.message }));
        process.exit(1);
    }
}

performOCR();
