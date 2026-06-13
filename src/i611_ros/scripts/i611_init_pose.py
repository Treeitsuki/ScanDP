import xmlrpc.client

server = xmlrpc.client.ServerProxy("http://192.168.0.23:4416/")

# init_joint = [90, 0, 90, -180, 90, 180]
# [445.82023659483554, -100.00070402799166, 295.7359466293506, 0.0, -0.0, -0.0]
# init_joint = [90, 15, 115, -180, 130, 180]
init_joint = [90, 50, 115, -180, 166, 180]
server.move_joint(init_joint)

# init_pose = [440, -100, 300, 0, 0, 0]
# server.move_position(init_pose)
