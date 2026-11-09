c = get_config()

# Redirect runtime dir away from broken Windows AppData directory
c.ServerApp.runtime_dir = r"C:\Users\nayan\.jupyter_runtime"
c.ServerApp.root_dir = r"D:\UROP"
