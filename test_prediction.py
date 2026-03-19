import os
from predict import predict_disaster


# -------------------------------
# CONFIG
# -------------------------------

TEST_IMAGE = "test.jpg"   # Put one test image in project folder


# -------------------------------
# MAIN TEST
# -------------------------------

def main():

    print("Starting model test...")

    # Check image exists
    if not os.path.exists(TEST_IMAGE):
        print("ERROR: Test image not found!")
        print(f"Place an image named '{TEST_IMAGE}' in project folder.")
        return

    try:
        # Run prediction
        result, confidence = predict_disaster(TEST_IMAGE)

        print("\n--- Prediction Result ---")
        print("Disaster Type :", result)
        print("Confidence    :", round(confidence * 100, 2), "%")

        print("\nTest completed successfully.")

    except Exception as e:

        print("\nERROR during prediction!")
        print(str(e))


# -------------------------------
# Run
# -------------------------------

if __name__ == "__main__":
    main()