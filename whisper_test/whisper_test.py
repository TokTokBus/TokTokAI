import whisper

# 음성 파일을 텍스트로 변환
def transcribe_audio(filename):
    model = whisper.load_model("base")
    result = model.transcribe(filename)
    return result["text"]

def extract_numbers(text):
    numbers = ''.join(filter(str.isdigit, text))
    return numbers

if __name__ == "__main__":
    # 음성을 텍스트로 변환
    transcription = transcribe_audio('1500번.m4a')

    # 변환된 텍스트 출력
    print("Original Transcription:", transcription)

    # 숫자만 추출
    numbers_only = extract_numbers(transcription)
    print("Numbers Only:", numbers_only)
