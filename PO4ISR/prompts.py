prompt_1_bundle = "1. Based on the user's current session interactions and specific details, please analyze the context and relevance of the candidate items to determine the user's preferences.\n"\
                "2. Consider their past interactions and interactive intent for each combination of items.\n"\
                "3. Reorder the 20 items in the candidate set based on the selected intent, taking into account the possibility of potential user interactions.\n"\
                "4. Provide the ranking results with the item index, ensuring that the order of all items in the candidate set is provided and that the items for ranking are within the candidate set."
    
prompt_2_bundle = "1. Analyze the session context and specific details of the interactions to identify combinations of items.\n"\
                   "2. Determine the user's interactive intent for each combination, taking into account their past interactions and preferences.\n"\
                   "3. Choose the intent that best represents the user's current needs and desires, considering the context and specific details of the interactions.\n"\
                   "4. Finally, reorder the 20 items in the candidate set based on the selected intent, taking into consideration the possibility of potential user interactions.\n"\
                   "Provide the ranking results with the item index, ensuring that the order of all items in the candidate set is provided."

prompt_1_ml = "Based on the user's current session interactions, please follow these steps to complete the subtasks:\n"\
            "1. Identify any patterns or commonalities among the items in the session interactions.\n"\
            "2. Consider the user's previous preferences and interactions to infer their current intent.\n"\
            "3. Evaluate the inferred intent and select the most likely representation of the user's preferences.\n"\
            "4. Utilize the selected intent to rank the items in the candidate set based on their relevance and potential user interest. Provide the ranking results with the item index.\n"\
            "Ensure that the order of all items in the candidate set is provided, and that the items for ranking are within the candidate set."
    
prompt_2_ml = "Given the user's current session interactions, please follow these steps to complete the subtasks:\n"\
            "1. Identify potential patterns or themes within the session by analyzing combinations of items.\n"\
            "2. Consider additional contextual information, such as genre, director, or actor, to better understand the user's interactive intent.\n"\
            "3. Evaluate the inferred intents alongside the user's previous ratings and preferences to determine the most accurate representation of their current preferences.\n"\
            "4. Utilize the selected intent to rank the 20 items in the candidate set based on the likelihood of user interactions. Provide the ranking results with the corresponding item index.\n"\
            "Ensure that the order of all items in the candidate set is provided, and that the items for ranking are within the candidate set."

prompt_1_games = "Your task is to analyze the user's current session interactions and the candidate set of items to accurately infer the user's preferences and intent.\n"\
                "1. Consider any patterns or combinations of items within the session that may indicate the user's genre preference or other relevant criteria.\n"\
                "2. Evaluate the context and relevance of the items in the candidate set to the user's session interactions, taking into account factors like genre, price, or other relevant criteria.\n"\
                "3. Deduce the user's interactive intent within each combination, considering factors like price comparison, genre preference, or other relevant criteria.\n"\
                "4. Rearrange the items in the candidate set based on the inferred intent, ensuring that the items are ordered according to the likelihood of potential user interactions.\n"\
                "Provide the rearranged list of items from the candidate set, along with their corresponding item indices."

prompt_2_games = "Based on the user's current session interactions and the candidate set of items, your task is to:\n"\
                "1. Identify any patterns or combinations of items within the session that may indicate the user's preferences or intent.\n"\
                "2. Evaluate the context and relevance of the items in the candidate set to the user's current session interactions and their overall preferences.\n"\
                "3. Deduce the user's interactive intent within each combination, considering factors like genre preference, price comparison, or other relevant criteria.\n"\
                "4. Rearrange the items in the candidate set based on the inferred intent, ensuring that the items are ordered according to the likelihood of potential user interactions.\n"\
                "Please provide the rearranged list of items from the candidate set, along with their corresponding item indices."