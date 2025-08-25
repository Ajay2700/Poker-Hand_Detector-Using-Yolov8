git init# Real-Time Poker Hand Detector

A real-time poker hand detection system using YOLOv8 and OpenCV that can recognize playing cards through a webcam and determine poker hand rankings.

## Features

- Real-time playing card detection using YOLOv8
- Support for all 52 cards in a standard deck
- Automatic poker hand evaluation
- GPU acceleration support for faster inference
- Detects and classifies the following poker hands:
  - Royal Flush
  - Straight Flush
  - Four of a Kind
  - Full House
  - Flush
  - Straight
  - Three of a Kind
  - Two Pair
  - Pair
  - High Card

## Requirements

- Python 3.8+
- PyTorch with CUDA support (for GPU acceleration)
- OpenCV-Python
- Ultralytics YOLOv8
- cvzone
- NumPy

## Installation

1. Clone this repository:

```bash
git clone https://github.com/yourusername/poker-hand-detector.git
cd poker-hand-detector
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

1. Make sure your webcam is connected and accessible.

2. Run the main detection script:

```bash
python Poker-Hand_Detector.py
```

3. Hold playing cards in front of your webcam. The system will:

   - Detect and identify individual cards
   - Show confidence scores for each detection
   - Display the detected poker hand when 5 cards are visible

4. Press 'q' to quit the application.

## Project Structure

- `Poker-Hand_Detector.py`: Main script for real-time detection and visualization
- `PokerHandFunction.py`: Contains the poker hand evaluation logic
- `playingCards.pt`: YOLOv8 model trained on playing card dataset
- `requirements.txt`: List of Python dependencies

## How It Works

1. **Card Detection**:

   - Uses a custom-trained YOLOv8 model to detect and classify playing cards
   - Each card is detected with its rank and suit (e.g., "AS" for Ace of Spades)

2. **Hand Evaluation**:

   - Once 5 cards are detected, the system evaluates the poker hand
   - Cards are processed to determine ranks and suits
   - Multiple hand combinations are checked in descending order of value
   - The highest possible hand is displayed on screen

3. **Real-time Processing**:
   - GPU acceleration for faster inference times
   - Continuous frame processing from webcam input
   - Visual feedback with bounding boxes and confidence scores

## Performance

- Average inference time: ~31-32ms per frame
- Processing breakdown:
  - Preprocess: ~1.5-2.0ms
  - Inference: ~31-32ms
  - Postprocess: ~2.5-3.5ms
- Real-time performance at ~30 FPS with GPU acceleration

## Limitations

- Requires good lighting conditions for accurate detection
- Best performance when cards are clearly visible and not overlapping
- May require GPU for optimal real-time performance

## Future Improvements

- Support for multiple simultaneous hands
- Improved detection accuracy in challenging lighting conditions
- Support for different card designs/decks
- Score tracking and game statistics
- Support for other card games

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLOv8 by Ultralytics
- OpenCV community
- PyTorch team
