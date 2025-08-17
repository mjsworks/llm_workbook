## Tokens
Tokens are individual unites passed into a model. in the early days we would train a model character by character.<br>
So, we would take a model and that would train on a series of individual characters. while prediction, it would predict the next possible character based on the one came before.
<br>
it had some benefits. We only have a limited number of alphabets. it was offering us a limited and managable vocab size for the inputs. However, it also came with a lot of challenges. It was asking so much from the model. the model has to understand the context along with all the intelligence associated with it.

## we went a bit further then.
So, we started training the models on words. so, for the input the words would go as inputs. so the input becomes so huges as there are so many different words and the outcome may be one of the other possible word. so, it was a good thing as it was not asking the model so much. the model had it easy. but the vocab becomes enormous.

## then around the time where GPT came in
GPT - Generative Pre-Trained Transformers created something in between. Rather than training the model on individual characters or needing them to learn each individual words, we could take chunks of letters. these chunks could sometime be a part of a word, sometimes be a complete word - these were called as tokens. they trained the model as a form of series of tokens. There are several benefits to this approach. it was easy for words where they have same beginning but the ending is different

if these sound so absurd, there is a tool that openai offers, where we can put the inputs and see the tokens that GPT makes for different models.

`https://platform.openai.com/tokenizer`

these tokenizations also include spaces with the token if required. it essentially means the beginning part of the word

Rule of thumb for english:

```
1 token ~ 4 characters
1 token ~ 0.75 words
1000 tokens ~ 750 words
```