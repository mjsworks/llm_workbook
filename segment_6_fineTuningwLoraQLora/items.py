"""
This file is a helper for the curation of the dataset.
We start by setting up a base model. Here that base model will be llama3.1-8b.
The reason being, we will be crafting our dataset so that it fits within a certain fixed number
of tokens as a maximum limit for the llama tokenizer. it is going to make it cheaper and easier
for us to train the model.
"""

## imports
from typing import Optional # indicates that certain fields can either have a value or be None
from transformers import AutoTokenizer # loads huggingface's tokenizer
import re # regular expressions

## configs
BASE_MODEL = "meta-llama/Llama-3.1-8B"
MIN_TOKENS = 150 # minimum tokens for dataset entry
MAX_TOKENS = 160 # maximum tokens for dataset entry
MIN_CHARS = 300 # minimum characters for dataset entry
CEILING_CHARS = MAX_TOKENS * 7 # truncates the content if this value is reached

class Item:
    """
    this is the main class for the single data entry
    """
    ## class level variables
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    PREFIX = "Price is $" # used when we build a prompt for the model
    QUESTION = "How much does it cost to the nearest dollar?" # used when we build a prompt for the model
    REMOVALS = ['"Batteries Included?": "No"','"Batteries Included?": "Yes"',
                '"Batteries Required?": "No"','"Batteries Required?": "Yes"',
                "By Manufacturer",
                "Item",
                "Date First",
                "Package",
                ":",
                "Number of",
                "Best Sellers",
                "Number",
                "Product"
    ] # list of unwanted phrases to strip from the product details. we dont want to waste tokens while training.

    ## attributes
    title: str
    price: float
    category: str
    token_count: int = 0
    details: Optional[str]
    prompt: Optional[str] = None # formatted text used for training
    include = False # a boolean flag to indicate if the item should be included in the dataset

    def __init__(self, data, price):
        self.title = data["title"]
        self.price = price
        self.parse(data)

    def scrub_details(self):
        """
        remove unnecessary phrases from the details string
        """
        details = self.details
        for remove in self.REMOVALS:
            details = details.replace(remove, "")
        return details
    
    def scrub(self, stuff):
        """
        clean up the provided text by removing unneccesary characters and whitespace
        also remove words that are 7+ characters and contain numbers, as they are likely to be product numbers

        uses regrex to remove special characters and whitespace
        splits into words and removes those longer than 6 letters and contains numbers
        """
        stuff = re.sub(r'[:\[\]"{}【】\s]+', ' ', stuff).strip()
        stuff = stuff.replace(" ,",",").replace(",,,", ",").replace(" ,,",",")
        words = stuff.split(' ')
        select = [word for word in words if len(word) < 7 or not any(char.isdigit() for char in word)]
        return " ".join(select)

    def parse(self, data):
        """
        gathers and scrubs description, features and details together
        truncates if too long, choecks minimum length
        encodes with the tokenizer to count tokens
        if enough tokens, decodes and builds a prompt with make_prompt()
        """
        contents = '\n'.join(data['description'])
        if contents:
            contents += '\n'
        features = '\n'.join(data['features'])
        if features:
            contents += features + '\n'
        self.details = data['details']
        if self.details:
            contents += self.scrub_details() + '\n'
        if len(contents)> MIN_CHARS:
            contents = contents[:CEILING_CHARS]
            text = f"{self.scrub(self.title)}\n {self.scrub(contents)}"
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) > MIN_TOKENS:
                tokens = tokens[:MAX_TOKENS]
                text = self.tokenizer.decode(tokens)
                self.make_prompt(text)
                self.include = True
    
    def make_prompt(self, text):
        """
        Set the prompt instance variable to be a prompt appropriate for training
        """
        self.prompt = f"{self.QUESTION}\n\n{text}\n\n"
        self.prompt += f"{self.PREFIX}{round(self.price)}.00\n\n"
        self.token_count = len(self.tokenizer.encode(self.prompt, add_special_tokens=False))

    def test_prompt(self):
        """
        Return a prompt suitable for testing, with the actual price removed
        """
        return self.prompt.split(self.PREFIX)[0] + self.PREFIX
    
    def __repr__(self):
        """
        return a String version of the item
        """
        return f"<{self.title} = ${self.price}>"
            
        