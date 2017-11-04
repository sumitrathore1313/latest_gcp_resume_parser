# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render

from rest_framework import generics, views
from .serializers import BucketlistSerializer
from .models import Bucketlist
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from rest_framework.response import Response
import subprocess
import json
from Helper import pdf2txt
from Helper import GramGloveSentenceVector
import os

class CreateView(generics.ListCreateAPIView):
    """This class defines the create behavior of our rest api."""
    queryset = Bucketlist.objects.all()
    serializer_class = BucketlistSerializer

    def perform_create(self, serializer):
        """Save the post data when creating a new bucketlist."""
        serializer.save()


class DetailsView(generics.RetrieveUpdateDestroyAPIView):
    """This class handles the http GET, PUT and DELETE requests."""

    queryset = Bucketlist.objects.all()
    serializer_class = BucketlistSerializer

class FileUploadView(views.APIView):
    parser_classes = (MultiPartParser, FormParser, JSONParser)

    def post(self, request,filename, format=None):
        my_file = request.FILES['file']

        with open('/home/sumitrathore1313/apps/django/django_projects/Project/data/'+filename, 'wb+') as temp_file:
            for chunk in my_file.chunks():
                temp_file.write(chunk)
        input_file = '/home/sumitrathore1313/apps/django/django_projects/Project/data/'+filename
        output_file = '/home/sumitrathore1313/apps/django/django_projects/Project/data/'+filename.split('.')[0]
        cmd = "java -cp 'bin/*:../GATEFiles/lib/*:../GATEFiles/bin/gate.jar:lib/*' code4goal.antony.resumeparser.ResumeParserProgram %s %s.json" % (input_file, output_file)
        os.chdir("/home/sumitrathore1313/apps/django/django_projects/Project/ResumeParser/ResumeTransducer/")
	subprocess.Popen(cmd, shell=True)
        import time
        time.sleep(20)
        with open('/home/sumitrathore1313/apps/django/django_projects/Project/data/'+filename.split('.')[0]+'.json') as json_data:
            data = json.load(json_data)

        return Response({'data': data})

class Registration(views.APIView):
    """docstring for Registration."""

    parser_classes = (MultiPartParser, FormParser, JSONParser)

    def post(self, request,filename ,format=None):

        my_file = request.FILES['file']

        with open('/home/sumitrathore1313/apps/django/django_projects/Project/data/'+filename, 'wb+') as temp_file:
            for chunk in my_file.chunks():
                temp_file.write(chunk)
        input_file = '/home/sumitrathore1313/apps/django/django_projects/Project/data/'+filename
        output_file = '/home/sumitrathore1313/apps/django/django_projects/Project/data/'+filename.split('.')[0]
        
	from keras.models import load_model
	classifier_path = "/home/sumitrathore1313/apps/django/django_projects/Project/data/Test/5GramClassifierOther"
	classifier = load_model(classifier_path)
	
	pdf2txt.convert(input_file, output_file)
	import re
	output_file = output_file+'.txt'
	with open(output_file, 'r') as f:
                content = f.readlines()
        import io
	
	with open(output_file, 'r') as f:
                resume_text = f.read().lower()
	gsv = GramGloveSentenceVector(output_file,dimension=50, training=False)
	sen2vec = gsv.get_5gram_sentenceVector()	
	import numpy as np
	y_pred = classifier.predict(np.array(sen2vec))	
	#return Response({"sen2vec": len(sen2vec), "dim": len(sen2vec[0])})
	count = 0
	labels = ['basic', 'experience', 'education', 'certificate', 'extra', 'skills', 'projects','summary', 'mimc']
	basic = []
	for i in range(5):
		basic.append(content[i].strip())
	for line in content:
    		if np.argmax(y_pred, 1)[count] == 0:
        		basic.append(line.strip())
    		count += 1
	from nltk.tag import StanfordNERTagger
	st = StanfordNERTagger('/home/sumitrathore1313/ner/ner/classifiers/english.all.3class.distsim.crf.ser.gz', '/home/sumitrathore1313/ner/ner/stanford-ner.jar')

	def getNER(data):
    		return st.tag(data.split())
	ner_basic = []
	for line in basic:
    		ner_basic.append(getNER(line))
	#return Response({'name': name, 'email': email, 'gender':gender, 'jobtitle':jobtitle})	
	temp = []
	name = []
	for i in range(len(ner_basic)):
    		for j in range(len(ner_basic[i])):
        		if ner_basic[i][j][1] == 'PERSON':
            			temp.append(ner_basic[i][j][0])
    		if len(temp):
        		name.append(" ".join(temp))
    		temp = []
	#return Response({'nname': name, "basic": ner_basic})
	number = ""
	email = ""	
	email_regex = re.compile(("([a-z0-9!#$%&'*+\/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+\/=?^_`"
                    "{|}~-]+)*(@|\sat\s)(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(\.|"
                    "\sdot\s))+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)"))
    
	number_regex = re.compile("(?:(?:\+?([1-9]|[0-9][0-9]|[0-9][0-9][0-9])\s*(?:[.-]\s*)?)?(?:\(\s*([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9])\s*\)|([0-9][1-9]|[0-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9]))\s*(?:[.-]\s*)?)?([2-9]1[02-9]|[2-9][02-9]1|[2-9][02-9]{2})\s*(?:[.-]\s*)?([0-9]{4})(?:\s*(?:#|x\.?|ext\.?|extension)\s*(\d+))?.*")
	eresult = re.search(email_regex, resume_text)	
	if eresult:
    		email = eresult.group(0)
	nresult = re.search(number_regex, resume_text)
	if nresult:
    		number = nresult.group(0)
	#return Response({'name': "hello"})	
	#number1 = unicode(number[0], errors='ignore')
	return Response({'name': name, 'email': email, 'phone':number})
