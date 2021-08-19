#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 20:07:47 2021

@author: pkjha
"""

import pickle
from get_data import getData
from math import isnan

def computeFraction( poi_messages, all_messages ):
    """ given a number messages to/from POI (numerator) 
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """


    ### you fill in this code, so that it returns either
    ###     the fraction of all messages to this person that come from POIs
    ###     or
    ###     the fraction of all messages from this person that are sent to POIs
    ### the same code can be used to compute either quantity

    ### beware of "NaN" when there is no known email address (and so
    ### no filled email features), and integer division!
    ### in case of poi_messages or all_messages having "NaN" value, return 0.
    
    poi_messages = float(poi_messages)
    all_messages = float(all_messages)
    
    if isnan(poi_messages) or isnan(all_messages):
        return 0.0
        
    fraction = poi_messages/all_messages
    return fraction


data_dict = getData() 

submit_dict = {}
for name in data_dict:
    print 'name: ', name
    data_point = data_dict[name]

    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
    print 'fraction_from_poi:', fraction_from_poi
    data_point["fraction_from_poi"] = fraction_from_poi


    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
    print 'fraction_to_poi: ', fraction_to_poi
    data_point["fraction_to_poi"] = fraction_to_poi
    
    submit_dict[name]={"from_poi_to_this_person":fraction_from_poi,
                       "from_this_person_to_poi":fraction_to_poi}
    
    
    
#####################

def submitDict():
    return submit_dict
