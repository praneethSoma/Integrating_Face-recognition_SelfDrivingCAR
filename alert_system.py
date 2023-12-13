import requests

# Function to send alerts via Telegram
def send_telegram_alert(bot_token, chat_id, message):
   
    send_message_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    data = {
        "chat_id": chat_id,
        "text": message,
    }
    response = requests.post(send_message_url, data=data)
    return response.json()


bot_token = "6938442518:AAHgxJ9Nb7kaHe0C5rS6OOhbUWt6b0h1Xe8"  
chat_id = "6377867755"  
message = "Unauthorized access detected! Please check immediately."

# Send the alert
result = send_telegram_alert(bot_token, chat_id, message)
#print(result)  