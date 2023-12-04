import openai
import random
import time

class Request():
    def __init__(self, config):
        self.conifg = config
        openai.api_key = self.conifg['openai_api_key']
    
    def request(self, user, system=None, message=None):
        response = self.openai_request(user, system, message)

        return response
    
    def openai_request(self, user, system=None, message=None):
        '''
        fix openai communicating error
        https://community.openai.com/t/openai-error-serviceunavailableerror-the-server-is-overloaded-or-not-ready-yet/32670/19
        '''
        if system:
            message=[{"role":"system", "content":system}, {"role": "user", "content": user}]
        else:
            content = system + user
            message=[{"role": "user", "content": content}]
        model = self.conifg['model']
        for delay_secs in (2**x for x in range(0, 10)):
            try:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages = message,
                    temperature=0.2,
                    frequency_penalty=0.0)
                break
            except openai.OpenAIError as e:
                randomness_collision_avoidance = random.randint(0, 1000) / 1000.0
                sleep_dur = delay_secs + randomness_collision_avoidance
                print(f"Error: {e}. Retrying in {round(sleep_dur, 2)} seconds.")
                time.sleep(sleep_dur)
                continue
        
        return response["choices"][0]["message"]["content"]
