#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

import numpy as np
import precice
import random

from helper import MBDynHelper, PreciceHelper
from input import MBDynInput

# create logger
module_logger = logging.getLogger('adapter')


class MBDynAdapter:
    def __init__(self, mbdyn_prep, config_file_name='../precice-config.xml',
                 participant_name=None, mesh_name=None, debugging=False,
                 inter_mesh=False, init_data=False, connectivity=False):

        if debugging:
            module_logger.setLevel(logging.DEBUG)
            log_file_handler = logging.FileHandler('adapter.log')
            log_file_handler.setLevel(logging.DEBUG)
            module_logger.addHandler(log_file_handler)
            print('Debugging enabled!')

        self.input = MBDynInput()
        self.input.create_from_prep(mbdyn_prep)

        self.mbdyn = MBDynHelper(mbdyn_prep.mesh)
        self.precice = PreciceHelper(config_file_name)

        if debugging:
            self.mbdyn._equidistant_samples()

        if 'internal pressure' in mbdyn_prep.nodes_dict.keys():
            self.mbdyn.set_pressure(mbdyn_prep.nodes_dict['internal pressure'])
            print('Internal Pressure set!')

        if not participant_name:
            participant_name = self.precice.get_participant_name_from_xml()

        self.participant_name = participant_name
        
        if self.participant_name== 'Structure_Solver':
        	if not mesh_name:
        	    #mesh_name = self.precice.get_mesh_name_from_xml()
        	    mesh_name = "Structure_Nodes" #Force & Displacement read & write
        else:
        	mesh_name = "Fluid_CellCenters" #Force write
        	mesh_name2 = "Fluid_Nodes" #Displacement read

        self.precice.setup_interface(solver_name=participant_name)

        if connectivity:
            self.precice.configure_interface(self.mbdyn.get_nodes(),
                                             grid_name=mesh_name,
                                             quads=mbdyn_prep.mesh.shells)
        else:
            self.precice.configure_interface(self.mbdyn.get_nodes(),
                                             grid_name=mesh_name)

        self.input.update_time_step(self.precice.time_step)
        self.input.write_input_file()
        self.mbdyn.initialize(case=mbdyn_prep.name)

        #TODO: testing
        self._init_data = init_data
        if self._init_data:
            if self.mbdyn.solve_initial():
                self.precice.displacement = self.mbdyn.get_absolute_displacement()
            else:
                raise ValueError('mbdyn failed to converge')
        self.precice.initialize_data()

    def run_simulation(self):
        iteration = 0
        previous_displacement =  0

        while self.precice.interface.is_coupling_ongoing():
            if self.precice.interface.is_action_required(
                    precice.action_write_iteration_checkpoint()):
                print("MBDyn Adapter: Writing iteration checkpoint")
                self.precice.interface.mark_action_fulfilled(
                    precice.action_write_iteration_checkpoint())

            self.precice.read_data()

            if self.precice.dimensions == 2:
                force_tensor = np.zeros((self.mbdyn.mesh.number_of_nodes(), 3))
                force_tensor[:, :self.precice.dimensions] = np.reshape(
                    self.precice.force, (-1, self.precice.dimensions))
            else:
                force_tensor = np.reshape(self.precice.force, (-1, 3))
                
            #import ipdb; ipdb.set_trace()
            force_tensor = np.random.uniform(low=0.4*10**-3,high=0.7*10**-6,size=(3,3))
            #entkoppelt von OpenFOAM input
            
            

            module_logger.debug('dt# = {}'.format(iteration))

            force_tensor_norm = np.linalg.norm(force_tensor, axis=1)
            module_logger.debug(
                'min, max, sum forces from precice:\n{}, {}, {}'.format(
                    np.min(force_tensor_norm), np.max(force_tensor_norm),
                    np.sum(force_tensor_norm)))
            module_logger.debug(
                'forces from precice sample:\n{}'.format(
                    force_tensor[self.mbdyn._debug_samples, :]))
            
            max_value_fluid = np.max(np.linalg.norm(force_tensor, axis=1))
            if max_value_fluid > 10:
                force_tensor = force_tensor / max_value_fluid * 0.3

            if self.mbdyn.load_changed:
                self.mbdyn.calc_pressure_forces(force_tensor)
            else:
                self.mbdyn.set_forces(force_tensor)

            if self.mbdyn.solve(False):
                module_logger.debug('Something went wrong!')
                break

            displacement = self.mbdyn.get_absolute_displacement()
            relative_displacement = displacement - previous_displacement

            module_logger.debug(
                'min, max relative displacement:\n{}, {}'.format(
                    np.min(relative_displacement),
                    np.max(relative_displacement)))
            module_logger.debug(
                'relative displacement sample:\n{}'.format(
                    relative_displacement[self.mbdyn._debug_samples, :]))

            #import ipdb; ipdb.set_trace()
            self.precice.write_data(relative_displacement)
            '''
            if self.participant_name== 'Structure_Solver':
            	self.precice.write_data(relative_displacement)
            else:
            	#import ipdb; ipdb.set_trace()
            	self.precice.write_data(self.precice.force.copy())
            '''

            self.precice.advance_time()

            if self.precice.interface.is_action_required(
                    precice.action_read_iteration_checkpoint()):
                print("MBDyn Adapter: Reading iteration checkpoint")
                self.precice.interface.mark_action_fulfilled(
                    precice.action_read_iteration_checkpoint())
            else:
                previous_displacement = displacement.copy()
                iteration = iteration + 1

                # self.mbdyn.set_forces(force_tensor)
                if self.mbdyn.solve(True):
                    break

                if (iteration % int(
                        self.input.control.entries['output frequency'])) == 0:
                    self.mbdyn.write_output_vtk(iteration)

        self.mbdyn.finalize()
        self.precice.interface.finalize()


if __name__ == "__main__":
    pass
