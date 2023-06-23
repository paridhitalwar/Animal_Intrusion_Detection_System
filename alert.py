from twilio.rest import TwilioRestClient as Call

From_Number = "+19782914922"
To_Number = "+919727838494"
Src_Path = "https://tarp1.000webhostapp.com/voice.xml"

def call():
    client = Call("AC64cbfd00f7647e8d3db0766951ac32f0","ecbe1d5f39de344cec097e73af89a057")
    print('Call initiated')
    client.calls.create(to=To_Number, from_=From_Number, url=Src_Path, method = 'Get')
    print('Call has been triggered successfully')

# print('Elephants have been detected on field.')

# from twilio.rest import Client

# # Your Account SID from twilio.com/console
# account_sid = "ACf97df9031a55cbd334904398f1f1cdfe"
# # Your Auth Token from twilio.com/console
# auth_token  = "7afa6dc75af550126ef98994345ab974"

# client = Client(account_sid, auth_token)

# message = client.messages.create(
#     to="+919727838494", 
#     from_="+14155238886",
#     body="Hello from Python!")

# print(message.sid)
# Download the helper library from https://www.twilio.com/docs/python/install
# import os
# from twilio.rest import Client


# # Find your Account SID and Auth Token at twilio.com/console
# # and set the environment variables. See http://twil.io/secure
# account_sid = os.environ['ACf97df9031a55cbd334904398f1f1cdfe']
# auth_token = os.environ['7afa6dc75af550126ef98994345ab974']
# client = Client(account_sid, auth_token)

# message = client.messages.create(
#                               body='Hello there!',
#                               from_='whatsapp:+14155238886',
#                               to='whatsapp:+919727838494'
#                           )

# print(message.sid)
