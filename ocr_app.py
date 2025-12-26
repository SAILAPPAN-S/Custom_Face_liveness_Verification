from flask import Flask, request, jsonify
from id_ocr import extract_text_from_image
from image_quality import validate_image_quality

app = Flask(__name__)

@app.route("/verify-id", methods=["POST"])
def verify_id():
    if "document" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["document"]

    # STEP 1: Image Quality Check
    quality_result = validate_image_quality(file)

    if quality_result["status"] == "REJECTED":
        return jsonify({
            "verification_status": "FAILED",
            "quality_check": quality_result
        }), 400

    # IMPORTANT: reset file pointer before OCR
    file.seek(0)

    # STEP 2: OCR
    ocr_result = extract_text_from_image(file)

    return jsonify({
        "verification_status": "PASSED",
        "quality_check": quality_result,
        "ocr_output": ocr_result
    })

if __name__ == "__main__":
    app.run(debug=True)
