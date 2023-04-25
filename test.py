from transformers import BlipProcessor, BlipForConditionalGeneration, Blip2Processor, \
            Blip2ForConditionalGeneration
            
print("loaded transformers!")

from main_simple_lib import *

im = load_image('https://wondermamas.com/wp-content/uploads/2020/04/IMG_8950-min-1024x1024.jpg')
print("image loaded!")

query = 'How many muffins can each kid have for it to be fair?'

print("calling get_code...")
code = get_code(query)

print("code: ",code)

execute_code(code, im, show_intermediate_steps=True)
