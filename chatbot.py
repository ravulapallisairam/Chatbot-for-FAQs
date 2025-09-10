# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 14:18:01 2025

@author: saira
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Our FAQ data: (Question, Answer) pairs
faq_data = [
    ("What is your return policy?", "You can return any item within 30 days."),
    ("How can I track my order?", "Use the tracking link in your confirmation email."),
    ("What payment methods do you accept?", "We accept credit card, debit card, UPI, and PayPal."),
    ("How do I contact customer support?", "Email us at support@example.com or call 123-456-7890.")
]

# Separate questions and answers
questions = [q for q, a in faq_data]
answers = [a for q, a in faq_data]

# Convert questions to vectors
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)

# Chatbot function
def chatbot():
    print("🤖 Chatbot: Hi! Ask me a question (type 'exit' to quit).")
    while True:
        user_input = input("You: ")

        if user_input.lower() == 'exit':
            print("🤖 Chatbot: Bye!")
            break

        user_vec = vectorizer.transform([user_input])
        similarity = cosine_similarity(user_vec, question_vectors)
        idx = similarity.argmax()

        if similarity[0][idx] < 0.3:
            print("🤖 Chatbot: Sorry, I don't understand your question.")
        else:
            print("🤖 Chatbot:", answers[idx])

# Run chatbot
chatbot()