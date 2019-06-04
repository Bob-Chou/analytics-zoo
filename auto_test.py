import time

from test.zoo.pipeline.onnx.test_model_loading import TestModelLoading as TML


if __name__ == "__main__":
    bot = TML()
    select_list = ["batch_norm", "slice", "concat", "unsqueeze", "squeeze", 
                   "test_mul", "div"]
    for func_name in dir(bot):
        if any([name in func_name for name in select_list]):
            print("Begin testing {}".format(func_name))
            in_time = time.time()
            getattr(bot, func_name)()
            print("finished in time {:.4f}s".format(time.time() - in_time))
            print()
            time.sleep(0.2)
