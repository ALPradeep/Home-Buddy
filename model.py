import pandas as pd
import tagui as t
import time
import random
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, AutoConfig, AutoModel

def lets_wait(target_element):
    while not t.present(target_element):
        t.wait(1)

    return target_element


def URL_generator(max_price, house_type, mrt_name):    
    df_MRT_code = pd.read_excel('DB\KB_MRT.xlsx',"Ave Rental",index_col = 0)
    mrt_name = mrt_name.lower()
    url_adapted = mrt_name.replace(" ","+")
    MRT_code = df_MRT_code['MRT ID'][mrt_name]
    split_code = MRT_code.split()

    url_mrt = ""
    concat_code = ""
    for code in split_code:
        url_mrt += f"&MRT_STATIONS[]={code}"
        concat_code += f"/{code}"
    concat_code = concat_code[1:]
    url_mrt = url_mrt[1:]
    
    ud = {'url':'https://www.propertyguru.com.sg/',
        'rental housing':'property-for-rent?market=residential&listing_type=rent&',
       'max price':f'maxprice={str(max_price)}&',
           'room':'beds[]=-1&',
           'entire house':'beds[]=0&beds[]=1&beds[]=2&beds[]=3&beds[]=4&beds[]=5&',
           'search true':'search=true',
           'MRT search method':f"{url_mrt}&freetext={concat_code}+{url_adapted}+MRT&",
           'sort method':'&sort=price&order=asc'}
    
    base = ud['url']
    rental_housing = ud['rental housing'] 
    max_price = ud['max price']
    house_type = ud[house_type.lower()]
    mrt_search = ud['MRT search method']
    ending = ud['search true']
    sort_method = ud['sort method']
    
    full_url = "".join([base,rental_housing,max_price,house_type,mrt_search,ending,sort_method])
    return full_url


def scraping_rule(rent_price, budget,results_size,results_threshold=20):
    if results_size > results_threshold:
        rent_price = float(rent_price)
        budget = float(budget)

        if rent_price<0.55*budget: # ignore properties too far lower
            scrape = False

        elif rent_price>0.55*budget and rent_price <0.7* budget:
            if random.random()<0.35: #scrape only 35% of these properties
                scrape = True
            else:
                scrape = False

        elif rent_price>0.7*budget and rent_price<0.8*budget:
            if random.random()<0.95: #scrape only 95% of these properties
                scrape = True
            else:
                scrape = False

        elif rent_price>0.8*budget and rent_price<=budget:
            if random.random()<0.55: #scrape only 55% of these properties
                scrape = True
            else:
                scrape = False

        else:
            scrape = False
    
    else:
        scrape = True
    
    return scrape


def input_info_extraction(INPUT):
    mrt_names = pd.read_excel('DB\KB_MRT.xlsx',"Ave Rental",index_col = 1)
    mrt_names = mrt_names['Station Name'].values.tolist()
    
    INPUT = INPUT.lower()

    # landmarks in INPUT, cp stands for checkpoint
    cp1 = INPUT.find('looking for')+11
    cp2 = INPUT.find('under')+5
    cp3 = INPUT[cp2:].find(',')+cp2
    cp4 = INPUT.find('mrt station')+9
    cp5 = INPUT.find('you prefer amenities')

    # house words
    room_words = ['room','rm','rom','shared apartment']
    house_words = ['house','apartment','HDB','home','hm']
    ovrall_housetype = room_words + house_words

    # extracting information
    try:
        budget = int(INPUT[cp2:cp3].replace('$',''))
    except Exception as e:
        print(e)

    for term in ovrall_housetype:
        if str(INPUT[cp1:cp2].find(term)).isnumeric() and term in room_words:
            house_type = 'room'
            break
        elif str(INPUT[cp1:cp2].find(term)).isnumeric() and term in house_words:
            house_type = 'entire house'
            break
        else:
            if term == ovrall_housetype[-1]:
                print(f"House type {term} in '{INPUT[cp1:cp2]}' is invalid")


    for mrt in mrt_names:
        if str(INPUT[cp4:cp5].find(mrt)).isnumeric():
            mrt_to_search = mrt
            break
        else:
            if mrt == mrt_names[-1]:
                print(f"MRT name in '{INPUT[cp4:cp5]}' is invalid")


    # generating URL            
    url = URL_generator(budget,house_type,mrt_to_search)
    print(f"""budget: {budget},
house_type: {house_type},
mrt_to_search: {mrt_to_search}""")
    print()

    print(f"URL: {url}")
    return url, budget


def description_scraping(url, budget):
    # Scraping Procedure
    start = time.time()

    wd = {'search box':'//input[@placeholder="Search Location"]',
        'prop title':'//div[@class="gallery-container"]/a/@title',
       'prop address':'//span[@itemprop="streetAddress"]',
        'prop price':'//li[@class="list-price pull-left"]/span[@class="price"]',
        'prop link':'(//div[@class="gallery-container"]/a/@href)',
        'next page':'//li[@class="pagination-next"]/a',
         'read more':'//a[@title="Click to view the rest of the details"]',
         'description':'//div[@itemprop="description"]',
         'stop code': '//span[@data-dobid="hdw"]'}

    # Initialisation
    prop_addresses, prop_prices, prop_descriptions, links = [[] for _ in range(4)]
    stop_words = ['  ','\\n','Description','Read More','*']
    next_page = True
    count = 1
    memory = []

    t.init()
    t.url(url)

    lets_wait(wd['search box'])

    results_size = t.read('//span[@class="shorten-search-summary-title"]')
    results_size = results_size.lower()

    if str(results_size.find('propert')).isnumeric():
        results_size = results_size[:results_size.find('propert')-1]
    elif str(results_size.find('result')).isnumeric():
        results_size = results_size[:results_size.find('result')-1]

    results_size = int(results_size.replace(',',''))

    while next_page:
        lets_wait(wd['prop address'])
        prop_no = t.count(wd['prop address'])
        for i in range(1, prop_no + 1):
            # Reading
            link = t.read(f"{wd['prop link']}[{i}]")
            prop_address = t.read(f"({wd['prop address']})[{i}]")
            price = t.read(f"({wd['prop price']})[{i}]")
            price = price.replace(',','')

            if prop_address not in memory and scraping_rule(price,budget,results_size)==True:            
                current_url = t.url()
                t.url(link)
                t.wait(2)

                if t.present(wd['read more']):
                    t.click(wd['read more'])

                description = t.read(lets_wait(wd['description']))
                for word in stop_words:
                    description = description.replace(word,'')

                # Appending
                prop_addresses.append(prop_address)
                prop_prices.append(price)
                links.append(link)
                memory.append(prop_address)
                prop_descriptions.append(description)

                t.url(current_url)
                t.wait(2)
            else:
                continue

        if t.present(wd['next page']):    
            count += 1
            t.url(f"{url[:49]}/{count}{url[49:]}")
            t.wait(3)

        else:
            next_page = False


    if (len(prop_addresses)+len(prop_prices)+len(prop_descriptions))/3 == len(links):
        dataset = pd.DataFrame({'Property' : prop_addresses,
                                  'Price (S$/mo)' : prop_prices,
                                  'Description': prop_descriptions,
                                 'Links' : links})
    else:
        dataset = "Lists do not tally"
    
    t.close()
    for i in range(len(dataset)):
        dataset['Description'][i] = dataset['Description'][i][1:-1]
    
    return dataset


def generate_embedding(input_text,model,tokenizer, save_as_numpy=False, max_length=512):
    encoded_input = tokenizer.encode_plus(
                input_text,
                add_special_tokens=True,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_token_type_ids=False,
                return_attention_mask=True,
                return_tensors='pt',
            )
    encoded_input.requires_grad = False
    last_hidden_state = model(encoded_input['input_ids'], attention_mask=encoded_input['attention_mask']).last_hidden_state


    # To get the [CLS] token representation (embedding), take the first token in the sequence
    cls_embedding = last_hidden_state[:, 0, :]
    if save_as_numpy:
        cls_embedding = cls_embedding.detach().numpy()
    return cls_embedding


def generate_embeddings(df, model, tokenizer):
    embeddings = []
    for b in range(len(df)):
        print(f"current batch: {b}")
        description = df.iloc[b].Description
        emb = generate_embedding(description,model=model,tokenizer=tokenizer, save_as_numpy=True)
        embeddings.append(emb)
    embeddings = np.vstack(embeddings)
    print(embeddings.shape)
    return embeddings


def expand_input(INPUT):
    remove_words = ["You're ","you're ", "You", "you", "as specific requests."]
    for word in remove_words:
        INPUT = INPUT.replace(word,"")
        
    INPUT = (INPUT + " ") * 3
    
    return INPUT


def generate_top_n_listing(emb,embeddings,n=3):
    similarities =  cosine_similarity(emb, embeddings)[0]
    # Get indices of top 3 most similar listings
    idxlist = list(np.argsort(similarities)[-n:])
    scorelist = [ similarities[i] for i in idxlist ]
    return idxlist,scorelist


def main(user_input, model,tokenizer,embeddings,n=3, display=False, df=None):    
    emb = generate_embedding(user_input,model=model, tokenizer=tokenizer, save_as_numpy=True)
    idxs,scores = generate_top_n_listing(emb,embeddings=embeddings,n=n)
#     print(idxs,scores)
    if display:
        print(f"user input:{user_input}\n")
        for i in range(n):
            print(f"################################## Listing {i+1}  ##################################")
            print("url:", df.iloc[idxs[i]].Links)
            description = df.iloc[idxs[i]].Description
            print(description)
    return idxs,scores


def language_model(user_input, scraped_dataset,
                  model_saving_path = "./model/BERT",
                   config_path = "./model/BERT_config",
                   tokenizer_saving_path = "./model/tokenizer"):
    
    # load the saved configuration
    config = AutoConfig.from_pretrained(config_path)

    # load the saved model weights into a new model instance
    model = AutoModel.from_pretrained(model_saving_path, config=config)

    tokenizer = BertTokenizer.from_pretrained(tokenizer_saving_path)
    
    df = scraped_dataset['Description']
    link = scraped_dataset['Links']
    
    embeddings_df = generate_embeddings(df=scraped_dataset,model=model, tokenizer=tokenizer)
    idxs,scores = main(user_input, model=model,tokenizer=tokenizer, embeddings=embeddings_df, df = scraped_dataset, display=True)
    
    final_shortlist = [link[idx] for idx in idxs]
    
    return final_shortlist


def run_process(INPUT):
    url, budget = input_info_extraction(INPUT)
    INPUT = expand_input(INPUT)
    scraped_dataset = description_scraping(url, budget)
    final_shortlist = language_model(INPUT, scraped_dataset)
    
    return final_shortlist


if __name__ == "main":
	run = False





