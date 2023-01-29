
class JobRunner():
    
        def __init__(self, model_instance_id):
            self.model_instance_id = model_instance_id

        def run(self):
    
            # Do some work
    
            self.job.status = Job.DONE
    
            self.job.save()