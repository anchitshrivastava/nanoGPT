import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer
from config import PROMPT_FORMAT, DEVICE

model = AutoModelForCausalLM.from_pretrained('mosaicml/mpt-1b-redpajama-200b-dolly', 
                                             trust_remote_code=True,
                                             cache_dir = "/home/sarabjot/storage/backup2/pathfactory_models")

tokenizer = AutoTokenizer.from_pretrained("mosaicml/mpt-1b-redpajama-200b-dolly")


def generate_response(instruction: str, *, model, tokenizer, 
                      do_sample: bool = True, max_new_tokens: int = 256, top_p: float = 0.92, top_k: int = 0, **kwargs) -> str:
    input_ids = tokenizer(PROMPT_FORMAT.format(instruction=instruction), return_tensors="pt").input_ids.to(DEVICE)

    # each of these is encoded to a single token
    response_key_token_id = tokenizer.encode("### Response:")[0]
    end_key_token_id = tokenizer.encode("### End")[0]

    gen_tokens = model.generate(input_ids, pad_token_id=tokenizer.pad_token_id, eos_token_id=end_key_token_id,
                                do_sample=do_sample, max_new_tokens=max_new_tokens, top_p=top_p, top_k=top_k, **kwargs)[0].cpu()
    print(tokenizer.decode(gen_tokens).strip())
    # find where the response begins
    response_positions = np.where(gen_tokens == response_key_token_id)[0]

    if len(response_positions) >= 0:
        response_pos = response_positions[0]
        
        # find where the response ends
        end_pos = None
        end_positions = np.where(gen_tokens == end_key_token_id)[0]
        if len(end_positions) > 0:
            end_pos = end_positions[0]

        return tokenizer.decode(gen_tokens[response_pos + 1 : end_pos]).strip()

    return None

prompt = """ Summarize the content given below.

3 Key Elements Of A Content Activation Strategy. A while back, we highlighted the perils of falling into a passive content trap. This post explores the antidote to passive content marketing and why some marketers are able to rise above it: It's called Content Activation. Content, data, and automation: common ingredients found in any basic B2B marketing cocktail. But what separates the good marketers from the truly great ones is a Content Activation Strategy. The world's highest-performing marketing teams are already activating their content and resources and seeing some pretty mind-blowing results. For example: Demandbase saw a 2X increase in responses to content promotions, a 3X boost in time visitors spent with content, and a 300% increase in influenced pipeline. So, what makes the Demandbase marketing team so different? Unlike passive content marketing, Content Activation never leaves the buyer's next step to chance. It's an adaptive process that delivers content in a more buyer-friendly way that encourages more efficient self-education to truly interested prospects. Shifting from passive to active content marketing requires a shift in mindset, model, and methodology to better reflect how buyers want to buy and removes obstacles, like forms and gates, commonly found in their way. What Content Activation is not In order to understand what Content Activation is, you first have to understand what it is not: Landing pages Sometimes, landing pages are necessary; however, they often act as a hard-stop in the buying journey and discourage your prospect from progressing. Premature forms If you ask for data before the prospect is ready, you'll scare them away. Only serve a form at peak interest when you're most likely to capture information. Blind clicks and downloads Knowing if an asset has been clicked on or downloaded is ok, but the real value comes from knowing if a lead actually spent time with it or not. Content Activation helps you never have to rely on blind vanity metrics and make every click or download the start of a journey. Content hubs Sure, content hubs are fine for casual browsing, but they can be detrimental for a truly goal-oriented buyer who is looking for specific information. Channel silos Having fragmented content experiences spread across too many channels creates an inconsistent buyer journey. Content activation amalgamates everything into a consistent, seamless content track, where each next step is clearly defined-regardless of the channel the prospect came from. One-size-fits-all content The more you know about your prospects, the better you can target and personalize. If you don't have that information you risk delivering impersonal, generic content experiences. 3 key elements of a Content Activation strategy Content Activation is a more proactive approach to your marketing strategy that guides prospects through their buyer's journey and prevents the pitfalls mentioned above. A solid Content Activation strategy involves these 3 key elements: 1. Adaptive content journeys For your content strategy to be truly activated you must be able to recommend a next best piece of content based on behavioral, engagement, and firmographic data. Content Activation allows your buyers' content journeys to be living, breathing things that can quickly and easily change course based on their behaviors. A Content Track is a prescribed series of content assets, each leading to the next, with the buyer's progress monitored, timed, and saved to their profile (for scoring and analyzing). For example, if you have a buyer persona that is focused on ABM, you can target them appropriately through multiple channels and guide them to a Content Track containing a collection of relevant content - blog posts, whitepapers, videos, third party articles, you name it! This way, you're not leaving anything 'to chance' and you give your prospects the opportunity to binge on your content while you have their attention. 2. A closed-loop approach Instead of a random browse through your content, Content Activation is a closed-loop approach that identifies the most engaged prospects-the ones who binge on content, progressing further along the content track-and guides your engagement strategy. It's a virtuous cycle of improvement in which deeper engagement with content leads to richer insight into each prospect's interests and intent; and that insight drives even deeper, more relevant engagements in the future. and so on. and so on. 3. Ability to spot Engaged Intent The major benefit of Content Activation is gaining access to a new class of data. Data like number of assets consumed in a single session and the amount of time spent on each asset. Deep metrics like this allow you to spot Engaged Intent-the leading indicator of marketing performance and sales-readiness. To learn more about Content Activation and the power of Engaged Intent check out this new eBook: An Introduction To Content Activation.
"""

print("*"*100)
print(generate_response(prompt, model=model, tokenizer=tokenizer))
print("*"*100)
print("hey")