"""
===========================================================
ERGAR Response Engine

Purpose:
Generate automated cyber response actions

Input:
Threat Risk Level

Output:
Response Action

===========================================================
"""


def generate_response(

        threat_level

):

    if threat_level == "High":

        return {

            "action":

                "BLOCK_AND_ISOLATE",


            "priority":

                "CRITICAL"

        }

    elif threat_level == "Medium":

        return {

            "action":

                "MONITOR_AND_LOG",


            "priority":

                "WARNING"

        }

    else:

        return {

            "action":

                "ALLOW",

            "priority":

                "NORMAL"

        }
