
from TimeExpandedNetwork import TimeExpandedNetwork
from dynamic_schedule_based_ue_model import dynamic_schedule_based_ue_model

def dynamic_schedule_based_ue_model_run():
	tn = TimeExpandedNetwork()
	od_flow = dynamic_schedule_based_ue_model(tn)