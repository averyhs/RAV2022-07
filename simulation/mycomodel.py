from cellsium.model import assemble_cell, SimulatedCell, h_to_s, s_to_h, RodShaped, BentRod
from cellsium.simulation.simulator import Timestep
from cellsium.model.agent import (
    Copyable,
    IdCounter,
    InitializeWithParameters,
    Representable,
    WithLineage,
    WithLineageHistory,
    WithRandomSequences,
    WithTemporalLineage,
)
from typing import Any, Iterable, Mapping, Optional
from random import randint
from math import cos, sin
from cmath import pi
import numpy as np

class MycoCellModel(SimulatedCell):
    # Class variables to be adjusted:
    # (TODO: should be adjusted from somewhere more accessible)
    div_time_mean = 3 # mean division time (hr)
    div_time_std = 0.3 # standard deviation of division time (hr)
    drug_intro_time = 6 # time (hr) at which drug is introduced
    drug_effect_tps = [0, 0.5, 1] # time points at which drug affects cells after drug intro
    drug_effect_prob = 0.1 # probability of a cell being affected by drug at each time pt (including already affected cells)
    long_div_time = 10 # longer div time mean (hr) due to drug

    @staticmethod
    def random_sequences(sequence):
        return dict(
            elongation_rate=sequence.normal(0.75, 0.09), # µm·h⁻¹
            div_time=sequence.normal(MycoCellModel.div_time_mean, MycoCellModel.div_time_std),
            long_div_time=sequence.normal(MycoCellModel.long_div_time, MycoCellModel.div_time_std),
            div_time_wiggle=sequence.normal(0, MycoCellModel.div_time_std),
            drug_dice=sequence.uniform()
            )

    def birth(self, parent=None, ts=None) -> None:
        self.elongation_rate = next(self.random.elongation_rate)
        
        # Inherit drug effects (actually inheriting division_time)
        # --------------------
        if parent is None:
            self.division_time = h_to_s(next(self.random.div_time))
        else:
            if round(ts.time)%7 <= 3: # mix it up sometimes (takes long)
                self.division_time = parent.division_time + h_to_s(next(self.random.div_time_wiggle))
            else:              
                self.division_time = parent.division_time
        # --------------------

    def grow(self, ts) -> None:
        self.length += self.elongation_rate * ts.hours

        # Drug perturbation
        # -----------------
        # (this might not be the best place to do this, but it runs every timestep so it will do for now)
        
        if ts.time in [h_to_s(self.drug_intro_time + x) for x in self.drug_effect_tps]:
            if next(self.random.drug_dice) <= self.drug_effect_prob:
                self.division_time = h_to_s(next(self.random.long_div_time))
        # -----------------

        if ts.time > (self.birth_time + self.division_time):
            offspring_a, offspring_b = self.divide(ts)
            offspring_a.length = offspring_b.length = self.length / 2
    
    def divide(self, ts: Timestep) -> Iterable["SimulatedCell"]:
        """
        Called when a cell should divide, creates the daughter cells.
        :param ts: Timestep
        :return: None
        """

        # 'v-snap' division behavior
        # --------------------------
        division_angle = randint(-45,45) * pi/180
        x, y = self.position
        alpha = self.angle
        length = self.length

        offspring_a, offspring_b = self.copy(), self.copy()

        offspring_a.position = [float(x-(length/4)*cos(alpha+division_angle)), float(y-(length/4)*sin(alpha+division_angle))]
        offspring_b.position = [float(x+(length/4)*cos(alpha)), float(y+(length/4)*sin(alpha))]
        offspring_a.angle = self.angle + division_angle
        offspring_b.angle = self.angle
        # --------------------------

        if isinstance(self, WithLineage):
            offspring_a.parent_id = offspring_b.parent_id = self.id_

        if isinstance(self, WithLineageHistory):
            offspring_a.lineage_history = self.lineage_history[:] + [self.id_]
            offspring_b.lineage_history = self.lineage_history[:] + [self.id_]

        if isinstance(self, WithTemporalLineage):
            now = ts.simulation.time
            offspring_b.birth_time = offspring_a.birth_time = now

        ts.simulator.add(offspring_a)
        ts.simulator.add(offspring_b)

        offspring_a.birth(parent=self, ts=ts)
        offspring_b.birth(parent=self, ts=ts)

        ts.simulator.remove(self)

        return offspring_a, offspring_b



Cell = assemble_cell(MycoCellModel, RodShaped)
