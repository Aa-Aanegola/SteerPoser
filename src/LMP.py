
import openai
from time import sleep
from openai._exceptions import RateLimitError, APIConnectionError
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import TerminalFormatter
from utils import load_prompt, DynamicObservation, IterableDynamicObservation
import time
from LLM_cache import DiskCache
from arguments import get_config
from steered_model import SteeredModel

class LMP:
    """Language Model Program (LMP), adopted from Code as Policies."""
    def __init__(self, name, cfg, fixed_vars, variable_vars, debug=False, env='rlbench', model=None):
        # fixed vars and variable vars define function interface available to LLM
        # fixed vars are static, come from transforms3d library
        # variable vars are VoxPoser defined functions, defined in interfaces.py.
        #  Includes access to other low-level LMPs, i.e. planner can call composer. 
        self._name = name # composer, planner, affordance map generator, etc
        self._cfg = cfg # cfg['lmp_config']['lmps'][name]
        self._debug = debug
        self._base_prompt = load_prompt(f"{env}/{self._cfg['prompt_fname']}.txt")
        self._stop_tokens = list(self._cfg['stop'])
        self._fixed_vars = fixed_vars
        self._variable_vars = variable_vars
        self.exec_hist = ''
        self._context = None
        self._cache = DiskCache(load_cache=self._cfg['load_cache'])
        
        
        if self._cfg['use_local']: # whether steering with local model vs openai api call
             if model is None:
                 self.steer_cfg = get_config(config_path='./configs/steering.yaml')
                 self.model = SteeredModel(self.steer_cfg)
             else:
                 print(f"using passed in model for LMP {self._name}")
                 self.model = model

    def clear_exec_hist(self):
        self.exec_hist = ''

    def build_prompt(self, query):
        if len(self._variable_vars) > 0:
            variable_vars_imports_str = f"from utils import {', '.join(self._variable_vars.keys())}"
        else:
            variable_vars_imports_str = ''
        prompt = self._base_prompt.replace('{variable_vars_imports}', variable_vars_imports_str)

        if self._cfg['maintain_session'] and self.exec_hist != '':
            prompt += f'\n{self.exec_hist}'
        
        prompt += '\n'  # separate prompted examples with the query part

        if self._cfg['include_context']:
            assert self._context is not None, 'context is None'
            prompt += f'\n{self._context}'

        user_query = f'{self._cfg["query_prefix"]}{query}{self._cfg["query_suffix"]}'
        prompt += f'\n{user_query}'

        return prompt, user_query
    
    def _cached_api_call(self, **kwargs):
        # check whether completion endpoint or chat endpoint is used
        if kwargs['model'] != 'gpt-3.5-turbo-instruct' and any([chat_model in kwargs['model'] for chat_model in ['gpt-3.5', 'gpt-4']]):
            # add special prompt for chat endpoint
            user1 = kwargs.pop('prompt')
            new_query = '# Query:' + user1.split('# Query:')[-1]
            user1 = ''.join(user1.split('# Query:')[:-1]).strip()
            user1 = f"I would like you to help me write Python code to control a robot arm operating in a tabletop environment. Please complete the code every time when I give you new query. Pay attention to appeared patterns in the given context code. Be thorough and thoughtful in your code. Do not include any import statement. Do not repeat my question. Do not provide any text explanation (comment in code is okay). I will first give you the context of the code below:\n\n```\n{user1}\n```\n\nNote that x is back to front, y is left to right, and z is bottom to up."
            assistant1 = f'Got it. I will complete what you give me next.'
            user2 = new_query
            # handle given context (this was written originally for completion endpoint)
            if user1.split('\n')[-4].startswith('objects = ['):
                obj_context = user1.split('\n')[-4]
                # remove obj_context from user1
                user1 = '\n'.join(user1.split('\n')[:-4]) + '\n' + '\n'.join(user1.split('\n')[-3:])
                # add obj_context to user2
                user2 = obj_context.strip() + '\n' + user2
            messages=[
                {"role": "system", "content": "You are a helpful assistant that pays attention to the user's instructions and writes good python code for operating a robot arm in a tabletop environment."},
                {"role": "user", "content": user1},
                {"role": "assistant", "content": assistant1},
                {"role": "user", "content": user2},
            ]
            kwargs['messages'] = messages
            # print kwargs to a file
            
            if kwargs in self._cache:
                print('(using cache)', end=' ')
                return self._cache[kwargs], kwargs
            else:
                response = openai.chat.completions.create(**kwargs)
                ret = response.choices[0].message.content
                # post processing
                ret = ret.replace('```', '').replace('python', '').strip()
                self._cache[kwargs] = ret
                return ret, kwargs
        else:
            if kwargs in self._cache:
                print('(using cache)', end=' ')
                return self._cache[kwargs], kwargs
            else:
                response = openai.completions.create(**kwargs)['choices'][0]['text'].strip()
                ret = response.choices[0].text.strip()
                self._cache[kwargs] = ret
                return ret, kwargs
    
    def _chat_template(self, messages):
        prompt = ''
        for message in messages:
            if message['role'] == 'user':
                prompt += f'{message["content"]}\n'
            elif message['role'] == 'assistant':
                prompt += f'{message["content"]}\n'
        return prompt
    
    def _local_call(self, **kwargs):
        user1 = kwargs.pop('prompt')
        new_query = '# Query:' + user1.split('# Query:')[-1]
        user1 = ''.join(user1.split('# Query:')[:-1]).strip()
        user1 = f"I would like you to help me write Python code to control a robot arm operating in a tabletop environment. Please complete the code every time when I give you new query. Pay attention to appeared patterns in the given context code. Be thorough and thoughtful in your code. Do not include any import statement. Do not repeat my question. Do not provide any text explanation (comment in code is okay). I will first give you the context of the code below:\n\n```\n{user1}\n```\n\nNote that x is back to front, y is left to right, and z is bottom to up. Your task is just to write clean and concise code that follows the template above based on the objects and query supplied below."
        user2 = new_query
        
        if user1.split('\n')[-4].startswith('objects = ['):
            obj_context = user1.split('\n')[-4]
            user1 = '\n'.join(user1.split('\n')[:-4]) + '\n' + '\n'.join(user1.split('\n')[-3:])
            user2 = obj_context.strip() + '\n' + user2
        messages=[
            {"role": "system", "content": "You are a helpful assistant that pays attention to the user's instructions and writes good python code for operating a robot arm in a tabletop environment."},
            {"role": "user", "content": user1},
            {"role": "user", "content": user2},
        ]
        kwargs['messages'] = messages

        kwargs['local'] = True

        # if kwargs in self._cache:
        #     print("using cached response yeehaw")
        #     return self._cache[kwargs], kwargs
        
        prompt = self._chat_template(messages)

        # print("prompting model with", prompt)
        
        ret = self.model.generate(prompt)

        self._cache[kwargs] = ret
        
        return ret, kwargs

    def __call__(self, query, **kwargs):
        prompt, user_query = self.build_prompt(query)

        start_time = time.time()
        while True:
            try:
                if self._cfg['use_local']:
                    code_str, _ = self._local_call(
                        prompt=prompt,
                        stop=self._stop_tokens,
                        temperature=self._cfg['temperature'],
                        max_tokens=self._cfg['max_tokens']
                    )
                    print(f'*** Local call took {time.time() - start_time:.2f}s ***')
                else:
                    code_str, _ = self._cached_api_call(
                        prompt=prompt,
                        stop=self._stop_tokens,
                        temperature=self._cfg['temperature'],
                        model=self._cfg['model'],
                        max_tokens=self._cfg['max_tokens']
                    )
                    print(f'*** OpenAI API call took {time.time() - start_time:.2f}s ***')
                break
            except (RateLimitError, APIConnectionError) as e:
                print(f'OpenAI API got err {e}')
                print('Retrying after 3s.')
                sleep(3)

        if self._cfg['include_context']:
            assert self._context is not None, 'context is None'
            to_exec = f'{self._context}\n{code_str}'
            to_log = f'{self._context}\n{user_query}\n{code_str}'
        else:
            to_exec = code_str
            to_log = f'{user_query}\n{to_exec}'

        to_log_pretty = highlight(to_log, PythonLexer(), TerminalFormatter())

        if self._cfg['include_context']:
            print('#'*40 + f'\n## "{self._name}" generated code\n' + f'## context: "{self._context}"\n' + '#'*40 + f'\n{to_log_pretty}\n')
        else:
            print('#'*40 + f'\n## "{self._name}" generated code\n' + '#'*40 + f'\n{to_log_pretty}\n')

        gvars = merge_dicts([self._fixed_vars, self._variable_vars])
        lvars = kwargs

        # return function instead of executing it so we can replan using latest obs（do not do this for high-level UIs)
        if not self._name in ['composer', 'planner']:
            to_exec = 'def ret_val():\n' + to_exec.replace('ret_val = ', 'return ')
            to_exec = to_exec.replace('\n', '\n    ')

        if self._debug:
            # only "execute" function performs actions in environment, so we comment it out
            action_str = ['execute(']
            try:
                for s in action_str:
                    exec_safe(to_exec.replace(s, f'# {s}'), gvars, lvars)
            except Exception as e:
                print(f'Error: {e}')
                import pdb ; pdb.set_trace()
        else:
            exec_safe(to_exec, gvars, lvars)

        self.exec_hist += f'\n{to_log.strip()}'

        if self._cfg['maintain_session']:
            self._variable_vars.update(lvars)

        if self._cfg['has_return']:
            if self._name == 'parse_query_obj':
                try:
                    # there may be multiple objects returned, but we also want them to be unevaluated functions so that we can access latest obs
                    return IterableDynamicObservation(lvars[self._cfg['return_val_name']])
                except AssertionError:
                    return DynamicObservation(lvars[self._cfg['return_val_name']])
            return lvars[self._cfg['return_val_name']]


def merge_dicts(dicts):
    return {
        k : v 
        for d in dicts
        for k, v in d.items()
    }
    

def exec_safe(code_str, gvars=None, lvars=None):
    print("Executing code now")
    banned_phrases = ['import', '__']
    for phrase in banned_phrases:
        print("running exec_safe for banned phrase:", phrase)
        assert phrase not in code_str, code_str
  
    if gvars is None:
        gvars = {}
    if lvars is None:
        lvars = {}
    empty_fn = lambda *args, **kwargs: None
    custom_gvars = merge_dicts([
        gvars,
        {'exec': empty_fn, 'eval': empty_fn}
    ])
    try:
        exec(code_str, custom_gvars, lvars)
    except Exception as e:
        print(f'Error executing code:\n{code_str}')
        raise e
    
