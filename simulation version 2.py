import datetime
import logging
import logging.config
import random
from dataclasses import dataclass
from datetime import timedelta
from typing import List
from tqdm import tqdm
import simpy
import yaml
from scipy.stats import truncnorm
import pandas as pd
import numpy as np

totes_distribution = pd.read_csv("sku_qty_distribution.csv", sep = ";")
totes_cdf = totes_distribution["cumulative_fraction"]
totes_pdf = totes_distribution["fraction"]

def useless_function(n):
    print(2*n)

def sample_skus():
    "uses the given cumulative distribution for the sku's to sample the nr of skus for one tote"

    return random.choices(totes_distribution["sku_quantity"], weights = totes_pdf)[0]


class EventLogger:
    """Logs events
    """

    def __init__(self, log_file_name: str):
        self.start_of_day = None

        with open("logging.conf") as logging_config_file:
            logging_config = yaml.load(logging_config_file, Loader=yaml.FullLoader)
        logging_config["handlers"]["file"]["filename"] = log_file_name
        logging.config.dictConfig(logging_config)
        self.__logger = logging.getLogger("Simulation")

        # Headers for the csv file:
        log_info = {
            "action": "action",
            "order_tote_id": "tote_batch_id",
            "timestamp": "timestamp",
            "location_id": "location"
        }
        self.__logger.info(msg="Log entry", extra=log_info)

    def log(self, action: str, order_tote_id: str, timestamp: int, location_id: str) -> None:
        """Logs an event in the output log.

        Args:
            action: The action of the event.
            order_tote_id: The tote id for the event.
            timestamp: The time of the event in milliseconds since start of day.
            location_id: The id of the station_id where the event occurs.

        """

        iso_timestamp = self.start_of_day + timedelta(milliseconds=timestamp)
        log_info = {
            "action": action,
            "order_tote_id": order_tote_id,
            "timestamp": str(iso_timestamp),
            "location_id": location_id
        }
        self.__logger.info(msg="Log entry", extra=log_info)


class PickPerformance(object):
    """Represents a pickperformance.

    Current implementation is a deterministic number of milliseconds per orderline.
    """

    def __init__(self, mean: int, st_dev: int, min_time: int, max_time: int) -> None:
        """Initialize a new pick performance distribution.

        Args:
            mean: The mean number of milliseconds for a pick.
            st_dev: The standard deviation in milliseconds for a pick.
            min_time: The minimal number of milliseconds for a pick.
            max_time: The maximum number of milliseconds for a pick.
        """

        self.__mean = mean
        self.__st_dev = st_dev
        self.__a = (min_time - mean) / st_dev
        self.__b = (max_time - mean) / st_dev

    def generate_picking_time(self, nr_samples: int = 1) -> List[int]:
        """Generate nr_samples picking times in milliseconds.

        Args:
            nr_samples: The number of picking times to generate.

        Returns:
            A list with nr_samples picking times in milliseconds.
        """

        samples = truncnorm.rvs(self.__a, self.__b, loc=self.__mean, scale=self.__st_dev, size=nr_samples)
        return [int(pick_time) for pick_time in samples]


@dataclass
class Configuration:
    """The environment elements of the manual pick zone.
    
    Args:
        env: The simpy environment.
        pickers: The simpy resource representing the pickers.
        consolidation_stations: The simpy resource representing the consolidation stations.
        activating_batches: The simpy.Store that holds activating batches.
        orderline_pick_performance: The pick peformance for picking orderlines.
        consolidation_performance: The performance per tote for consolidation stations.
    """

    env: simpy.Environment
    pickers: simpy.Resource
    lanes: simpy.Resource
    consolidation_stations: simpy.Resource
    activating_batches: simpy.Store
    orderline_pick_performance: PickPerformance
    consolidation_performance: PickPerformance


class ManualPickZone:
    """Represents the manual pick zone.
    
    Assumptions/implementation:
    1. Unlimited number of buffer lanes.
    2. No travel time from lane to consolidation station.
    3. Single queue for consolidation stations, cart goes to first free consolidation station.
    4. All batches are equal:
        1. No priority.
        2. All have equal max_batch_size.
    """
    
    def __init__(
            self,
            nr_of_lanes : int,
            nr_of_pickers: int,
            max_batch_size: int,
            nr_of_consolidation_stations: int,
            max_waiting_time: int,
            env: simpy.Environment,
            logger: EventLogger
    ):

        """

        Args:
            nr_of_pickers: The number of order pickers (max nr of active carts).
            max_batch_size: The max batch size.
            nr_of_consolidation_stations: The number of consolidation stations.
            max_waiting_time: The maximum time from the first tote arriving in a batch until the batch is released for picking.
            env: The simpy simulation environment.
            logger: The event logger.
        """
        self.__nr_of_lanes = nr_of_lanes # Number of lanes / buffers
        self.__nr_of_carts = nr_of_pickers  # Max number of active pickers.
        self.max_batch_size = max_batch_size  # Max number of totes on a cart.
        self.__nr_of_consolidation_stations = nr_of_consolidation_stations  # Number of consolidation stations
        self.__max_waiting_time = max_waiting_time  # Max nr of seconds tote is allows to wait before batch has to start picking.
        self.__next_batch_id = 0
        self.__logger = logger
        self.__accepting_batches = dict()
        self.__active_batches = dict()
        self.__configuration = Configuration(
            env=env,
            pickers=simpy.Resource(env=env, capacity=self.__nr_of_carts),
            lanes = simpy.Resource(env=env, capacity=self.__nr_of_lanes),
            consolidation_stations=simpy.Resource(env=env, capacity=self.__nr_of_consolidation_stations),
            activating_batches=simpy.Store(env=env),
            orderline_pick_performance=PickPerformance(mean=18000, st_dev=3000, min_time=6000, max_time=30000),
            consolidation_performance=PickPerformance(mean=6000, st_dev=1000, min_time=3000, max_time=9000),
        )
        self.__configuration.env.process(self.process_activating_batches())
        
    def handle_tote(self, tote_id: str, nr_skus: int) -> simpy.Event:
        tote_finish_event = self.__configuration.env.event()

        with self.__configuration.lanes.request() as lane_request:
                yield lane_request
                self.__logger.log(
                    action="batch-pick-start",
                    order_tote_id="batch-" + str(self.batch_id),
                    timestamp=self.__configuration.env.now,
                    location_id="buffer"
                )






        if len(self.__accepting_batches) > 0:
            # There are batches accepting totes.
            oldest_batch_id = list(self.__accepting_batches.keys())[0]
            oldest_batch = self.__accepting_batches[oldest_batch_id]
            if oldest_batch.accepts_new_tote():
                oldest_batch.add_tote(tote_id=tote_id, nr_skus=nr_skus, tote_finish_event=tote_finish_event)
            else:
                raise Exception("Cannot add tote " + tote_id + " to accepting batch " + str(oldest_batch.batch_id) + ".")
        else:
            # wait for a free lane, then create new batch

            with self.__configuration.lanes.request() as lane_request:
                yield lane_request
                self.__logger.log(
                    action="batch-pick-start",
                    order_tote_id="batch-" + str(self.batch_id),
                    timestamp=self.__configuration.env.now,
                    location_id="buffer"
                )





            # Create a new batch
            batch = Batch(
                batch_id=self.__next_batch_id,
                zone_configuration=self.__configuration,
                max_batch_size=self.max_batch_size,
                max_waiting_time=self.__max_waiting_time,
                logger=self.__logger
            )
            self.__logger.log(
                action="new-batch",
                order_tote_id="batch-" + str(batch.batch_id),
                timestamp=self.__configuration.env.now,
                location_id="buffer"
            )
            batch.add_tote(tote_id=tote_id, nr_skus=nr_skus, tote_finish_event=tote_finish_event)
            self.__accepting_batches[self.__next_batch_id] = batch
            self.__configuration.env.process(batch.process())
            self.__next_batch_id += 1
        return tote_finish_event
    
    def process_activating_batches(self):
        """Register a batch as active (released for picking) instead of accepting.
        
        Returns: An event generator.

        """
        
        while True:
            activating_batch_id = yield self.__configuration.activating_batches.get()  # wait for a batch to be release for picking.
            if activating_batch_id in self.__accepting_batches:
                batch = self.__accepting_batches[activating_batch_id]
                self.__accepting_batches.pop(activating_batch_id)
                self.__active_batches[activating_batch_id] = batch


class Batch:
    """Represents a batch of totes in the manual pick zone."""
    
    def __init__(self, batch_id: int, zone_configuration: Configuration, max_batch_size: int, max_waiting_time: int, logger: EventLogger):
    
        self.batch_id = batch_id
        self.__configuration = zone_configuration
        self.__max_batch_size = max_batch_size
        self.__max_waiting_time = max_waiting_time
        self.__tote_to_finish_event = dict()  # Mapping from tote_id to tote_finished_slow events.
        self.__tote_to_nr_skus = dict()
        self.__batch_full_event = self.__configuration.env.event()
        self.__released = False
        self.__logger = logger
        

    def accepts_new_tote(self) -> bool:
        """Determine if this batch accepts new totes.
        
        Returns: True if this batch has not yet started picking and the max number of totes for this batch is not exceeded.

        """
        
        return (not self.__released) and (len(self.__tote_to_finish_event) < self.__max_batch_size)
    
    def add_tote(self, tote_id: str, nr_skus: int, tote_finish_event: simpy.Event):
        """Add tote to this batch and create a tote finished event that will be triggered by this batch.
        
        Args:
            nr_skus: The nr of skus to pick for the tote.
            tote_id: The tote id of the tote to add to this batch.
            tote_finish_event: The event to trigger when the tote is finished in slow.

        Returns: The tote_finished_event for the tote to add to this batch.

        """
        
        if not self.accepts_new_tote():
            raise Exception("Tote " + tote_id + " is added to batch " + str(self.batch_id) + " while it is not allowed.")
        self.__tote_to_finish_event[tote_id] = tote_finish_event
        self.__tote_to_nr_skus[tote_id] = nr_skus
        self.__logger.log(
            action="add-to-batch_" + str(self.batch_id),
            order_tote_id=tote_id,
            timestamp=self.__configuration.env.now,
            location_id="buffer"
        )
        if len(self.__tote_to_finish_event) == self.__max_batch_size:
            self.__batch_full_event.succeed()
    
    def process(self):
    
        # Wait until the max waiting time has been exceeded or the batch has reached the maximum batch size.
        batch_timeout_event = self.__configuration.env.timeout(self.__max_waiting_time)
        result = yield self.__batch_full_event | batch_timeout_event

        # Batch has been released for picking, do not accept new totes anymore.
        self.__configuration.activating_batches.put(self.batch_id)
        self.__released = True

        # Check if timeout is the triggered event:
        if batch_timeout_event in result:
            self.__logger.log(
                action="batch-timeout",
                order_tote_id="batch-" + str(self.batch_id),
                timestamp=self.__configuration.env.now,
                location_id="buffer"
            )
        else:
            self.__logger.log(
                action="batch-full",
                order_tote_id="batch-" + str(self.batch_id),
                timestamp=self.__configuration.env.now,
                location_id="buffer"
            )

        # Log batch release
        self.__logger.log(
            action="batch-release",
            order_tote_id="batch-" + str(self.batch_id),
            timestamp=self.__configuration.env.now,
            location_id="buffer"
        )

        # Wait for an available picker.
        with self.__configuration.pickers.request() as picker_request:
            yield picker_request
            self.__logger.log(
                action="batch-pick-start",
                order_tote_id="batch-" + str(self.batch_id),
                timestamp=self.__configuration.env.now,
                location_id="buffer"
            )

            # Determine pick time of the cart depending on number of orderlines.
            total_nr_skus = 0
            for tote_id in self.__tote_to_nr_skus:
                total_nr_skus += self.__tote_to_nr_skus[tote_id]

            pick_times = self.__configuration.orderline_pick_performance.generate_picking_time(nr_samples=total_nr_skus)
            total_pick_time = sum(pick_times)

            # Pick the items.
            yield self.__configuration.env.timeout(total_pick_time)

            self.__logger.log(
                action="batch-pick-end",
                order_tote_id="batch-" + str(self.batch_id),
                timestamp=self.__configuration.env.now,
                location_id="buffer"
            )
            # Exit 'with' clause to free up the picker to pick another batch.

        # Wait for an available consolidation station.
        with self.__configuration.consolidation_stations.request() as consolidation_station_request:
            yield consolidation_station_request

            self.__logger.log(
                action="batch-consolidate-start",
                order_tote_id="batch-" + str(self.batch_id),
                timestamp=self.__configuration.env.now,
                location_id="consolidation"
            )

            # Start consolidation.
            nr_totes = len(self.__tote_to_finish_event)
            consolidation_times = self.__configuration.consolidation_performance.generate_picking_time(nr_samples=nr_totes)
            for (tote_id, consolidation_time) in zip(self.__tote_to_finish_event, consolidation_times):
                yield self.__configuration.env.timeout(consolidation_time)
                self.__tote_to_finish_event[tote_id].succeed()

            self.__logger.log(
                action="batch-consolidate-end",
                order_tote_id="batch-" + str(self.batch_id),
                timestamp=self.__configuration.env.now,
                location_id="consolidation"
            )


class Tote:
    """Represents a tote
    """

    def __init__(self, tote_id: str, nr_skus: int, arrival_time: int, manual_pick_zone: ManualPickZone, progress_bar: tqdm, env: simpy.Environment, logger: EventLogger):
        """

        Args:
            tote_id: The tote id.
            nr_skus: The number of skus this tote requires from the manual pick zone.
            arrival_time: The arrival time of this tote, in miliseconds since start of day.
            manual_pick_zone: The manual pick zone instance.
            progress_bar: The progress bar showing the number of finished totes.
            env: The simpy environment.
            logger: The event logger.
            
        """

        self.__tote_id = tote_id
        self.__nr_skus = nr_skus
        self.__arrival_time = arrival_time
        self.__manual_pick_zone = manual_pick_zone
        self.__progress_bar = progress_bar
        self.__env = env
        self.__logger = logger
        

    def process(self):

        # Wait until this tote arrives
        yield self.__env.timeout(self.__arrival_time)

        # Wait until the manual pick zone has finished handling this tote
        yield self.__manual_pick_zone.handle_tote(tote_id=self.__tote_id, nr_skus=self.__nr_skus)

        # Log finishing the tote, as this tote may be finished earlier than another tote in the same batch.
        self.__logger.log(
            action="tote-finished",
            order_tote_id=self.__tote_id,
            timestamp=self.__env.now,
            location_id="consolidation"
        )

        sojourn_times[int(self.__tote_id) - 1]= self.__env.now - self.__arrival_time
        self.__progress_bar.update(1)


# ==================================================== Start simulation here ===========================================================================


if __name__ == "__main__":

    random.seed(7858363)  # Use random seed for reproducable results.

    # Initialize logger, environment and manual pick zone
    logger = EventLogger(log_file_name="manual_pick_zone.csv")
    logger.start_of_day = datetime.datetime.now().replace(hour=6, minute=0, second=0, microsecond=0)  # set start of day to today at 6:00.
    env = simpy.Environment()
    manual_pick_zone = ManualPickZone(
        nr_of_pickers=20,
        max_batch_size=12,
        max_waiting_time=600000,  # 600000 milisec = 10 minutes
        nr_of_consolidation_stations=8,
        env=env,
        logger=logger
    )

    # TODO generate totes here, now done with random start times and only 1 sku.

    nr_totes = 1000
    sojourn_times = np.ones(nr_totes)
    progress_bar = tqdm(total=nr_totes)
    for i in range(0, nr_totes):
        tote = Tote(
            tote_id= str(i),
            nr_skus= sample_skus(),
            arrival_time=random.randint(0, 28800000),  # random nr of milisec with at most 8 hours
            manual_pick_zone=manual_pick_zone,
            env=env,
            progress_bar=progress_bar,
            logger=logger
        )
        env.process(tote.process())  # Register the tote process in the simulation environment!

    # Run the simulation
    env.run()
    progress_bar.close()


import matplotlib.pyplot as plt
plt.hist(sojourn_times/60000)
plt.xlabel('Sojourn times')
plt.show()