import threading
import os

x=0

def process():
    global x 
    print("process")
    x+=1

def thread_task(lock):
    """
    This function process each inference
    """
    print("Task assigned to thread: {}".format(threading.current_thread().name)) 
    print("ID of process running task: {}".format(os.getpid())) 
    
    lock.acquire()
    process() #process
    lock.release()

def main():
    global x
    x=0
    lock=threading.Lock()

    t1=threading.Thread(target=thread_task,args=(lock,),name='[T1]')
    t2=threading.Thread(target=thread_task,args=(lock,),name='[T2]')
    t3=threading.Thread(target=thread_task,args=(lock,),name='[T3]')

    t1.start()
    t2.start()
    t3.start()

    t1.join()
    t2.join()
    t3.join()


if __name__=="__main__":
    for i in range(10):
        main()
        print (i,x)
