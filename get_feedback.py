# get_feedback.py

def generate_feedback(user_answer: str, correct_answer: str) -> str:
    user = user_answer.strip()
    correct = correct_answer.strip()

    if not user:
        return "回答が入力されていません。まずは考えて書いてみましょう！"
    
    if user == correct:
        return "正解です！非常によくできました！"
    
    if correct in user:
        return "一部は正しい内容が含まれていますが、表現やキーワードが足りません。もう一歩です！"

    if len(user) < 5:
        return "短すぎる回答です。もう少し具体的に書いてみましょう。"

    return "残念ながら正解とは異なります。もう一度、問題文をよく読んで考えてみましょう。"
