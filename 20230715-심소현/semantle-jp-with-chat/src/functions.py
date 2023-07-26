guess_word = {"name": "guess_word",
                  "description": "Use this function to check if a guessed word is the correct answer or not, and if incorrect, calculate a score and a rank of the guess word.",
                  "parameters": {
                      "type": "object",
                      "properties": {
                          "word": {
                              "type": "string",
                              "description": "A single Japanese word to guess, which is can be a noun, verb, adverb or adjective. e.g. 空, 近い, 行く, etc."
                              },
                        #   "puzzle_num": {
                        #       "type": "integer",
                        #       "description": "An index indicating today's puzzle."
                        #   }
                      },
                      "required": ["word"]
                  }}

lookup_answer = {"name": "lookup_answer",
                "description": "Use this function to check the correct answer of today's puzzle.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        # "puzzle_num": {
                        #     "type": "integer",
                        #     "description": "An index indicating today's puzzle."
                        }
                    # },
                    # "required": ["puzzle_num"]
                }}

retrieve_puzzle_num = {"name": "retrieve_puzzle_num",
                  "description": "Use this function to retrieve today's puzzle number.",
                  "parameters": {
                      "type": "object",
                      "properties": {}
                  },
                  }

update_history = {"name": "update_history",
                  "description": "Use this function to add current guess to a table for a user's guess history.",
                  "parameters": {
                      "type": "object",
                      "properties": {
                          "current_guess": {
                              "type": "json",
                              "description": "A currently guessed word and its score and rank."
                          },
                          "guess_history": {
                              "type": "object",
                              "description": "A dataframe containing the guessed words and its score and rank in a row."
                          }
                      },
                      "required": ["current_guess", "guess_history"]
                  }}
read_rule = {"name": "read_rule",
                 "description": "Use this function to read the game rule for clarification of your response.",
                 "parameters": {
                     "type": "object",
                     "properties": {},
                 }}


def get_functions():
    functions = [guess_word, 
                 lookup_answer, 
                #  retrieve_puzzle_num, 
                #  update_history, 
                 read_rule]
    return functions