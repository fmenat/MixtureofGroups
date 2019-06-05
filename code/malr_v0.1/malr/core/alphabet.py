# -*- coding: utf-8 -*-
# Copyright (C) 2012, Filipe Rodrigues <fmpr@dei.uc.pt>

class Alphabet:
	
	def __init__(self):
		self._index_dict = {}
		self._object_dict = {}
		self._next_index = 0
	
	def add(self, obj):
		if self._object_dict.has_key(obj):
			return None
		self._index_dict[self._next_index] = obj
		self._object_dict[obj] = self._next_index
		self._next_index += 1
		return obj
		
	def lookup_index(self, index):
		return self._index_dict[index]
		
	def lookup_object(self, obj):
		return self._object_dict[obj]
		
	def __len__(self):
		return self._next_index
	
	def get_objects(self):
		return self._object_dict.keys()
	