import threading
import os
from time import sleep

x=0

def record(thread):
    global x 
    print("{} RECORD".format(thread))
    sleep(1.1)
    x+=1

def process(thread):
    global x 
    print("{} PROCESS".format(thread))
    sleep(1.2)
    x+=1

def playing(thread):
    global x 
    print("{} PLAYING".format(thread))
    sleep(1.1)
    x+=1

def thread_task(lock_record,lock_playing):
    """
    This function process each inference
    """
    # print("Task assigned to thread: {}".format(threading.current_thread().name)) 
    # print("ID of process running task: {}".format(os.getpid())) 
    thread=threading.current_thread().name
    # lock.acquire()
    lock_record.acquire()
    record(thread)
    lock_record.release()
    process(thread) #process
    lock_playing.acquire()
    playing(thread)
    lock_playing.release()
    # lock.release()

def main():
    global x
    x=0

    lock_record=threading.Lock()
    lock_playing=threading.Lock()

    t1=threading.Thread(target=thread_task,args=(lock_record,lock_playing,),name='[T1]')
    t2=threading.Thread(target=thread_task,args=(lock_record,lock_playing,),name='[T2]')
    t3=threading.Thread(target=thread_task,args=(lock_record,lock_playing,),name='[T3]')

    t1.start()
    t2.start()
    t3.start()

    t1.join()
    t2.join()
    t3.join()


if __name__=="__main__":
    for i in range(3):
        main()
        print (i,x)
