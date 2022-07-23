from pythainlp.tokenize import word_tokenize
def wangchan_ner_result_to_extracted_answer(result,original_text):
        new_result = [x for x in result if x["entity_group"]!="O"]

        entities = []
        tmp_entity = ""
        for elem in new_result:
            if elem["entity_group"][0]=="B" or elem["entity_group"][0]=="O":
                if tmp_entity!="":
                    entities+=[tmp_entity]
                    tmp_entity=elem["word"].strip() if elem["word"] in original_text else (elem["word"][0].upper()+elem["word"][1:]).strip()
                else:
                    tmp_entity+=elem["word"]
            else :
                tmp_entity+=elem["word"]

        if tmp_entity!="":
            entities+=[tmp_entity]

        entities = [entity.replace(" ","").replace("<_>"," ")  for entity in entities ]
        entities = [entity.strip() if entity in original_text else (entity[0].upper()+entity[1:]).strip() for entity in entities ]
            
        return entities

def preprocess_for_wangchan(text):
    return "".join(word_tokenize(text.lower().replace(" ","<_>")))