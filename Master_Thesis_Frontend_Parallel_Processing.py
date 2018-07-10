# -*- coding: utf-8 -*-
#!/usr/bin/env python3

from flask import Flask , render_template , request , send_file
import os
import shutil
import multiprocessing
import time
import queue # imported for using queue.Empty exception
from zipfile import ZipFile
from disambiguation import disambiguate


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.abspath("static/upload_folder")
app.config['DOWNLOAD_FOLDER'] = os.path.abspath("static/download_folder")
ALLOWED_EXTENSIONS = set(['txt'])
# files_all ---->  filename ----> {'original':[True/False,{'PERSON':False,'LOCATION':False}]}
# true false for if the the file has passed thru allowed_file() , person and location file names
files_all = {}
# list of running processes for person , location and both states
running_processes = {'person':[],'location':[],'both':[]}
# maximum allowed concurent processes
MAX_PROCESSES = multiprocessing.cpu_count()

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/",methods=['GET'])
def index():
	global running_processes
	global files_all
	global task_to_be_done

	# clearing upload folder
	if os.path.exists(app.config['UPLOAD_FOLDER']):
		shutil.rmtree(app.config['UPLOAD_FOLDER'])
	os.mkdir(app.config['UPLOAD_FOLDER'])
	# clearing download folder
	if os.path.exists(app.config['DOWNLOAD_FOLDER']):
		shutil.rmtree(app.config['DOWNLOAD_FOLDER'])
	os.mkdir(app.config['DOWNLOAD_FOLDER'])

	files_all = {}
	# terminating all running processes if any running
	terminate_and_clear_process_list(task_to_be_done,"", running_processes)

	template = 'index.html'
	return render_template(template)

@app.route("/uploaded", methods = ['POST'])
def upload_file():
	global files_all
	global running_processes
	if request.method == 'POST':
		# terminating all running processes if any running
		terminate_and_clear_process_list(task_to_be_done,"", running_processes)
		files = request.files.getlist('files')
		if files:
			# saving all valid files in upload folder
			for rec in files:
				if allowed_file(rec.filename):
					rec.save(os.path.join(app.config['UPLOAD_FOLDER'], rec.filename))
					files_all[rec.filename]=[True,{'PERSON':False,'LOCATION':False}]
				else:
					files_all[rec.filename]=[False,{'PERSON':False,'LOCATION':False}]

			file_list =[[key,value[0]] for key, value in files_all.items()]
			template = 'uploaded.html'
			return render_template(template,file_list=file_list)
		else:
			return '''
			<!doctype html>
				<h1 align='center'>No files Uploaded</h1>
				<div style='text-align:center'><button onclick="window.history.back()">GO BACK</button></div>
			'''

@app.route("/disambiguation", methods = ['GET'])
def disambiguation_processing():
	state = False
	process_done = 0
	global files_all
	global running_processes
	global task_to_be_done
	easy_files_format_list = []
	if request.method == 'GET':
		template = 'disambiguation.html'

		if request.args.get('action',False) == 'person':
			state = 'person'

			if not running_processes[state]:
				terminate_and_clear_process_list(task_to_be_done,state, running_processes)
				start_process(task_to_be_done,'PERSON', state, files_all, running_processes)

			if running_processes[state]:
				process_done = join_process(task_to_be_done,state, running_processes)
				files_all = get_files_all_status(files_all,state,["PERSON"]) # a list of file and process status
				easy_files_format_list = convert_format(files_all)
				if process_done==1:
					create_zip_file(state,"PERSON",files_all)
			#print ("----running process list---",running_processes)
			return render_template(template,file_status=easy_files_format_list,state=state,process_done=process_done)

		if request.args.get('action',False) == 'location':
			state = 'location'
			if not running_processes[state]:
				terminate_and_clear_process_list(task_to_be_done,state, running_processes)
				start_process(task_to_be_done,'LOCATION', state, files_all, running_processes)

			if running_processes[state]:
				process_done = join_process(task_to_be_done,state, running_processes)
				files_all = get_files_all_status(files_all,state,["LOCATION"]) # a list of file and process status
				easy_files_format_list = convert_format(files_all)
				if process_done==1:
					create_zip_file(state,"LOCATION",files_all)
			return render_template(template,file_status=easy_files_format_list,state=state,process_done=process_done)

		'''
		if request.args.get('action',False) == 'both':
			state = 'both'
			if not running_processes[state]:
				terminate_and_clear_process_list(state, running_processes)
				start_process( 'PERSON', state, files_all, running_processes)
				start_process( 'LOCATION', state, files_all, running_processes)

			if running_processes[state]:
				process_done = join_process(state, running_processes)
				files_all = get_files_all_status(files_all,running_processes,state,['PERSON','LOCATION']) # a list of file and process status
				easy_files_format_list = convert_format(files_all)
			return render_template(template,file_status=easy_files_format_list,state=state,process_done=process_done)
		'''
		state = False

		return '''
			<!doctype html>
				<h1 align='center'>Nothing to see here</h1>
				<div style='text-align:center'><button onclick="window.history.back()">GO BACK</button></div>
			'''

		#template = 'uploaded.html'
		#return render_template(template)

@app.route('/return-files/person')
def return_files_person():
	download_directory = app.config['DOWNLOAD_FOLDER']
	try:
		return send_file(os.path.join(download_directory,"person.zip"), attachment_filename='person.zip')
	except Exception as e:
		return str(e)


@app.route('/return-files/location')
def return_files_location():
	download_directory = app.config['DOWNLOAD_FOLDER']
	try:
		return send_file(os.path.join(download_directory,'location.zip'), attachment_filename='location.zip')
	except Exception as e:
		return str(e)


def create_zip_file(state,tag,files_all):
	download_directory = app.config['DOWNLOAD_FOLDER']
	all_download_files=os.listdir(download_directory)
	with ZipFile(os.path.join(download_directory,state+'.zip'), 'w') as myzip:
		for key,value in files_all.items():
			if value[0] and (tag+"_"+key) in all_download_files:
				myzip.write(os.path.join(download_directory,tag+"_"+key),tag+"_"+key)


def convert_format(files_all):
	# formatlist is [filename,True/False,person,location]
	format_list=[]
	for key,value in files_all.items():
		#if not value[0] or value[1]['PERSON'] or value[1]['LOCATION']:
		format_list.append([key,value[0],value[1]['PERSON'],value[1]['LOCATION']])
	return format_list

def get_files_all_status(files_all,state,tags_search=[]):
	all_download_files=os.listdir(app.config['DOWNLOAD_FOLDER'])
	for filename,value in files_all.items():
		if value[0]:
			for tag in tags_search:
				if (tag+"_"+filename) in all_download_files:
					files_all[filename][1][tag]=tag+"_"+filename
	return files_all



def start_process(task_to_be_done, tag_search,state,file_list,running_processes):
	for filename,value in file_list.items():
		task_to_be_done.put(filename)
	for rec in range(MAX_PROCESSES):
		j = multiprocessing.Process(target=worker, name="process_"+str(rec+1), args=(task_to_be_done,app.config['UPLOAD_FOLDER'],app.config['DOWNLOAD_FOLDER'],tag_search))
		j.start()
		print (j.name)
		running_processes[state].append(j)

def join_process(task_to_be_done,state,running_processes):
	process_done=0
	if task_to_be_done.empty():
		process_done=1
		for j in running_processes[state]:
			j.join()
	return process_done

def terminate_and_clear_process_list(task_to_be_done,not_state,running_processes):
	# emptying queue
	while True:
		try:
			task_to_be_done.get_nowait()
		except queue.Empty:
			break

	states = ['person','location','both']
	states = list(set(states)-set([not_state]))
	for rec in states:
		if running_processes[rec]:
			for j in running_processes[rec]:
				j.terminate()
				j.join()
			running_processes[rec]=[]
	# before starting new processes clearing the download folder
	if os.path.exists(app.config['DOWNLOAD_FOLDER']):
		shutil.rmtree(app.config['DOWNLOAD_FOLDER'])
	os.mkdir(app.config['DOWNLOAD_FOLDER'])
	return True


def worker(task_to_be_done,main_directory,download_directory,tag_search):
	name = multiprocessing.current_process().name
	while True:
		try:
			text_document_name = task_to_be_done.get_nowait()
			print ('Now running:{} with task {}'.format(name,text_document_name))
			disambiguate(text_document_name, main_directory,download_directory,tag_search)
			#time.sleep(4)
			print('Now exiting:{} with task {}'.format(name,text_document_name))
		except queue.Empty:
			break
	return True



if __name__ == '__main__':
	task_to_be_done = multiprocessing.Queue()
	app.run(host="0.0.0.0",port=5000,debug=True,use_reloader=True)

	pass
