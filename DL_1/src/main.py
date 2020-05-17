import pattern
import NumpyTests
import generator
# x_checker = pattern.Checker(100,25)
# x_circle = pattern.Circle(10,2,(5,5))
#
# # x_checker.draw()
# # x_checker.show()
# x_circle.draw()
# x_circle.show()

file_path ='./exercise_data/'
label_path = 'Labels.json'
batch_size = 12
image_size = [32,32,3]
rotation =True
mirroring =False
shuffle=False
z1 = generator.ImageGenerator(file_path, label_path, batch_size, image_size, rotation, mirroring, shuffle)
b1 = z1.next()
z1.show()
b2 = z1.next()
z1.show()

# x = NumpyTests.TestGen()
#
# # x.setUp()
# # # x.testInit()
# # # x.testResetIndex()
# # x.testResize()
# # # x.testMirroring()
# # x.testRotation()
# # z1.show()


