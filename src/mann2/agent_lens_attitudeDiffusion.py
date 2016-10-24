from os import environ
from subprocess import call
import random
from sys import getsizeof

from pandas import DataFrame

from mann2 import agent_lens_recurrent
from mann2.utils.tail import tail


class AgentLensAttitudeDiffusion(agent_lens_recurrent.AgentLensRecurrent):
    num_class_insances = 0

    def __init__(self, config, logger):
        """Create AgentLensAttitudeDiffusion agent.

        Parameters
        ----------
        config: dict
            configurations, usually a yaml file that has been loaded by python into a dict.

            configuration params needed:
                num_banks, bank_length, bank_init_value, bank_names
        logger: logging handle
        """
        super(AgentLensAttitudeDiffusion, self).__init__()

        self._agent_id = AgentLensAttitudeDiffusion.num_class_insances
        AgentLensAttitudeDiffusion.num_class_insances += 1

        self.config = config
        self.logger = logger

        self.logger.debug('config: {}'.format(self.config))

        self.num_banks = self.config['agent']['num_banks']
        self.bank_length = self.config['agent']['bank_length']
        self.bank_init_value = self.config['agent']['bank_init_value']
        self.bank_names = self.config['agent']['bank_names']

        assert self.num_banks == len(self.bank_names)

        self.state_length = self.num_banks * self.bank_length

        self._state = DataFrame({self.bank_names[x]: [self.bank_init_value] *
                                 self.bank_length for x in range(self.num_banks)})

        self._prototype = self.get_prototype()
        self.past_states_for_write = {}


    def __repr__(self):
        return '{}-{} STATE:\n{}'.format(
            self.__class__.__name__,
            self.agent_id,
            self.state)

    def get_prototype(self):
        pos = self.config['lens']['prototype']['pos']
        neg = self.config['lens']['prototype']['neg']
        self.logger.debug('pos prototype: {}'.format(pos))
        self.logger.debug('neg prototype: {}'.format(neg))
        p = pos + neg
        self.logger.debug('Prototype: {}'.format(p))
        return(p)

    def _write_ex_file_training(self, filepath):
        self.logger.debug('Writing example training file: {}'.format(filepath))
        with open(filepath, 'w') as f:
            self.logger.debug('Writing training example: {}'.format(filepath))
            for example in range(self.config['lens']['training']['num_examples']):
                # TODO for mutated prototypes, you can write the code here
                ex_string = 'name:proto{}\nB: {};\n'.format(
                    example,
                    ' '.join(str(x) for x in self.prototype))
                self.logger.debug('Writing example string: {}'.format(ex_string))
                f.write(ex_string)

    def _write_ex_file_seed(self, filepath):
        with open(filepath, 'w') as f:
            self.logger.debug('Writing example seed file: {}'.format(filepath))
            ex_string = 'name: seed\nB: {};\n'.format(
                ' '.join(str(x) for x in self.prototype))
            self.logger.debug('Writing example string: {}'.format(ex_string))
            f.write(ex_string)

    def _write_ex_file_sim(self, filepath):
        with open(filepath, 'w') as f:
            self.logger.debug('Agent {} is writing an ex file in sim mode'.format(self.agent_id))
            ex_string = 'name: agent_{}\nB: {};\n'.format(
                self.agent_id,
                self.state_str(' ', ' '))
            self.logger.debug('Writing ex: {}'.format(ex_string))
            f.write(ex_string)

    def write_ex_file(self, filepath, mode, **kwargs):
        """Write an example file for lens

        mode can be 'state', 'training', 'prototype'
        """
        self.logger.debug('Writing ex file: {} with mode: {}'.format(filepath, mode))
        if mode == 'training':
            self._write_ex_file_training(filepath)
        elif mode == 'seed':
            self._write_ex_file_seed(filepath)
        elif mode == 'sim':
            self._write_ex_file_sim(filepath)
        else:
            raise ValueError('Unknown mode for write_ex_file')

    def append_agent_id_to_base_fn(self, base_fn, f_ext):
        self.logger.debug('Appending agentID to base filename: {}'.format(base_fn))
        self.logger.debug('file_extension: {}'.format(f_ext))
        new_filename = base_fn.replace(f_ext, '_{}' + f_ext)
        self.logger.debug('New filename string replacement: {}'.format(new_filename))
        new_filename = new_filename.format(self.agent_id)
        self.logger.debug('New filename with agentID: {}'.format(new_filename))
        return(new_filename)

    def train(self, in_file):
        self.logger.info('Training agent: {}'.format(self))
        example_file = self.config['lens']['training']['filename']['fn']
        self.logger.debug('Example file from config: {}'.format(example_file))

        if self.config['lens']['training']['filename']['append_agent_id']:
            example_file = self.append_agent_id_to_base_fn(example_file, '.ex')

        self.write_ex_file(example_file, 'training')

        # values in lens_env need to be a str
        self.call_lens(in_file, lens_env={'mann_a': str(self.agent_id),
                                          'mann_exfile': example_file})
        return(self)

    def write_state_to_f(self, tick, value, output_f_agent_step_info):
        self.logger.debug('Writing state to file: {}'.format(output_f_agent_step_info))
        output_f_agent_step_info.write('{},{},{},{}\n'.format(tick, None,
                                                            self.agent_id,
                                                            value))

    def store_write_states(self, tick, output_f_agent_step_info):
        self.logger.debug('Storing agent: {}, state num: {}, total size: {}'.format(
            self.agent_id, tick, len(self.past_states_for_write)))

        self.past_states_for_write[tick] = self.state_str(',', ',')

        if getsizeof(self.past_states_for_write) >= self.config['single_sim']['agent_write_size']:
            self.logger.debug('State size threshold limit reached, writing to output...')
            self.logger.debug('Current past states for write: {}'.format(self.past_states_for_write.keys()))
            for time, state in self.past_states_for_write.items():
                self.write_state_to_f(time, state, output_f_agent_step_info)
            self.past_states_for_write = {}

    def _update_seed(self, tick, in_file, ex_file, out_file, split_index):
        self.logger.info('Updating agent in seed mode')
        self.write_ex_file(ex_file, 'seed')
        self.call_lens(in_file,
                       lens_env={
                           'mann_a': str(self.agent_id),
                           'mann_ex_base_fn': 'seed_cycle'}) # TODO hard-coded 'seed_cycle' base name
        self.logger.debug('pre-update state:\n{}'.format(self.state))
        self.state = self.get_new_state_from_outfile(out_file, split_index)
        self.logger.debug('post-update state:\n{}'.format(self.state))

    def _update_sim_random_1(self, tick, in_file, ex_file, out_file, split_index, graph):
        self.logger.debug('Updating agent in sim mode')

        predecessor_ids = graph.predecessors(self.agent_id)
        self.logger.debug('Predecessors: {}'.format(predecessor_ids))

        if predecessor_ids == []:
            self.logger.debug('No Predecessors for agent: {}'.format(self.agent_id))
            return(None)
        else:
            selected_predecessor = random.sample(predecessor_ids, 1)[0]
            self.logger.debug('Selected predecessor: {}'.format(selected_predecessor))
            predecessor_agent = graph.node[selected_predecessor]['agent']
            self.logger.debug('Selected Agent: {}'.format(predecessor_agent))
            predecessor_agent.write_ex_file(ex_file, 'sim')
            self.call_lens(in_file,
                           lens_env={
                               'mann_a': str(self.agent_id),
                               'mann_ex_base_fn': 'sim'}) # TODO hard-coded 'sim'
            self.logger.debug('pre-update state:\n{}'.format(self.state))
            self.state = self.get_new_state_from_outfile(out_file, split_index)
            self.logger.debug('post-update state:\n{}'.format(self.state))

    def update(self, mode,
               tick,
               in_file,
               ex_file, ex_file_append_id,
               out_file, out_file_append_id,
               output_f_agent_step_info=None,
               graph=None,
               **mode_kwargs):
        if ex_file_append_id:
            ex_file = self.append_agent_id_to_base_fn(ex_file, '.ex')
        if out_file_append_id:
            out_file = self.append_agent_id_to_base_fn(out_file, '.out')

        if mode == 'seed':
            self._update_seed(tick, in_file, ex_file, out_file, 0)  # TODO
        elif mode == 'random_1':
            self._update_sim_random_1(tick, in_file, ex_file, out_file, 0, graph)
        else:
            raise ValueError('Unknown update mode: {}'.format(mode))

        self.store_write_states(tick, output_f_agent_step_info)

    def get_new_state_from_outfile(self, outfile, split_index):
        self.logger.debug('Getting new state from: {}'.format(outfile))
        with open(outfile, 'r') as f:
            output_lines = tail(f)
        out_df = DataFrame({'outfile': output_lines})
        out_df['new_state'] = out_df.outfile.str.split(' ', expand=True)[split_index]
        self.logger.debug('out_df:\n{}'.format(out_df))

        pos = out_df.new_state[ :self.bank_length].reset_index(drop=True)
        neg = out_df.new_state[self.bank_length: ].reset_index(drop=True)
        self.logger.debug('pos:\n{}'.format(pos))
        self.logger.debug('neg:\n{}'.format(neg))

        new_state = DataFrame({'pos': pos, 'neg': neg})
        self.logger.debug('new_state:\n{}'.format(new_state))
        return(new_state)

    def call_lens(self, in_file, lens_env):
        self.logger.info('Calling Lens (Python)')
        self.logger.debug('Using in file: {}'.format(in_file))
        self.logger.debug('Lens env: {}'.format(lens_env))

        env = environ  # from os
        self.logger.debug('env: {}\n\n'.format(env))

        env.update(lens_env)

        self.logger.debug('env with lens_env: {}\n\n'.format(env))

        self.logger.debug('Lens subprocess call')
        call(['lens', '-batch', in_file], env=env)  # from subprocess
        self.logger.debug('Lens call finished (Python)')

    def state_str(self, state_delim, pos_neg_delim):
        pos_str = state_delim.join([str(x) for x in self.state.pos])
        neg_str = state_delim.join([str(x) for x in self.state.neg])
        state_str = pos_neg_delim.join([pos_str, neg_str])
        return(state_str)

    @property
    def agent_id(self):
        return self._agent_id

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, new_state):
        assert isinstance(new_state, DataFrame)
        assert new_state.shape[0] * new_state.shape[1] == self.state_length
        self._state = new_state

    @property
    def prototype(self):
        return self._prototype
    @prototype.setter
    def prototype(self, new_prototype):
        assert isinstance(new_prototype, list)
        assert len(new_prototype) == self.state_length
        self.logger.debug('Prototype set: {}'.format(new_prototype))
        self._prototype = new_prototype
