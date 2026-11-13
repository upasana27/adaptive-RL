from gym_cooking.cooking_world.world_objects import *
from gym_cooking.cooking_book.recipe import Recipe, RecipeNode
from copy import deepcopy


def id_num_generator():
    num = 0
    while True:
        yield num
        num += 1


id_generator = id_num_generator()

#  Basic food Items
# root_type, id_num, parent=None, conditions=None, contains=None
ChoppedLettuce = RecipeNode(root_type=Lettuce, id_num=next(id_generator), name="Lettuce",
                            conditions=[("chop_state", ChopFoodStates.CHOPPED)])
ChoppedOnion = RecipeNode(root_type=Onion, id_num=next(id_generator), name="Onion",
                          conditions=[("chop_state", ChopFoodStates.CHOPPED)])
ChoppedTomato = RecipeNode(root_type=Tomato, id_num=next(id_generator), name="Tomato",
                           conditions=[("chop_state", ChopFoodStates.CHOPPED)])
ChoppedCarrot = RecipeNode(root_type=Carrot, id_num=next(id_generator), name="Carrot",
                          conditions=[("chop_state", ChopFoodStates.CHOPPED)])
MashedCarrot = RecipeNode(root_type=Carrot, id_num=next(id_generator), name="Carrot",
                          conditions=[("blend_state", BlenderFoodStates.MASHED)])
ChoppedPotato = RecipeNode(root_type=Potato, id_num=next(id_generator), name="Potato",
                           conditions=[("chop_state", ChopFoodStates.CHOPPED)])
ChoppedBroccoli = RecipeNode(root_type=Broccoli, id_num=next(id_generator), name="Broccoli",
                           conditions=[("chop_state", ChopFoodStates.CHOPPED)])

# Salad Plates
LettuceSaladPlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                              contains=[ChoppedLettuce])
TomatoSaladPlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                              contains=[ChoppedTomato])
ChoppedOnionPlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                              contains=[ChoppedOnion])
ChoppedCarrotPlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                         contains=[ChoppedCarrot])
MasedCarrotPlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                         contains=[MashedCarrot])
ChoppedPotatoPlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                                contains=[ChoppedPotato])
ChoppedBroccoliPlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                                contains=[ChoppedBroccoli])

TomatoLettucePlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                                contains=[TomatoSaladPlate, ChoppedLettuce])
LettuceTomatoPlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                                contains=[LettuceSaladPlate, ChoppedTomato])
TomatoOnionPlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                              contains=[TomatoSaladPlate, ChoppedOnion])
OnionTomatoPlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                                contains=[ChoppedOnionPlate, ChoppedTomato])
TomatoCarrotPlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                                contains=[TomatoSaladPlate, ChoppedCarrot])
CarrotTomatoPlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                                contains=[ChoppedCarrotPlate, ChoppedTomato])
LettuceOnionPlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                                contains=[LettuceSaladPlate, ChoppedOnion])
OnionLettucePlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                                contains=[ChoppedOnionPlate, ChoppedLettuce])
LettuceCarrotPlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                                contains=[LettuceSaladPlate, ChoppedCarrot])
CarrotLettucePlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                                contains=[ChoppedCarrotPlate, ChoppedLettuce])
OnionCarrotPlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                                contains=[ChoppedOnionPlate, ChoppedCarrot])
CarrotOnionPlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                                contains=[ChoppedCarrotPlate, ChoppedOnion])
TomatoPotatoPlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                               contains=[TomatoSaladPlate, ChoppedPotato])
PotatoTomatoPlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                               contains=[ChoppedPotatoPlate, ChoppedTomato])
LettucePotatoPlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                                contains=[LettuceSaladPlate, ChoppedPotato])
PotatoLettucePlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                                contains=[ChoppedPotatoPlate, ChoppedLettuce])
OnionPotatoPlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                              contains=[ChoppedOnionPlate, ChoppedPotato])
PotatoOnionPlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                              contains=[ChoppedPotatoPlate, ChoppedOnion])
CarrotPotatoPlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                               contains=[ChoppedCarrotPlate, ChoppedPotato])
PotatoCarrotPlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                               contains=[ChoppedPotatoPlate, ChoppedCarrot])
TomatoBroccoliPlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                               contains=[TomatoSaladPlate, ChoppedBroccoli])
BroccoliTomatoPlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                               contains=[ChoppedBroccoliPlate, ChoppedTomato])
LettuceBroccoliPlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                                contains=[LettuceSaladPlate, ChoppedBroccoli])
BroccoliLettucePlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                                contains=[ChoppedBroccoliPlate, ChoppedLettuce])
OnionBroccoliPlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                              contains=[ChoppedOnionPlate, ChoppedBroccoli])
BroccoliOnionPlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                              contains=[ChoppedBroccoliPlate, ChoppedOnion])
CarrotBroccoliPlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                               contains=[ChoppedCarrotPlate, ChoppedBroccoli])
BroccoliCarrotPlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                               contains=[ChoppedBroccoliPlate, ChoppedCarrot])
PotatoBroccoliPlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                               contains=[ChoppedPotatoPlate, ChoppedBroccoli])
BroccoliPotatoPlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                               contains=[ChoppedBroccoliPlate, ChoppedPotato])

TomatoLettuceOnionPlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                                     contains=[ChoppedTomato, ChoppedLettuce, ChoppedOnion])

# 4-ingredient plates
# R2: Lettuce + Tomato + Onion + Carrot (build from 2-ingredient bases)
LettuceTomatoOnionCarrotPlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                                           contains=[LettuceTomatoPlate, ChoppedOnion, ChoppedCarrot])

# R3: Lettuce + Potato + Onion + Broccoli
LettucePotatoOnionBroccoliPlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                                             contains=[LettucePotatoPlate, ChoppedOnion, ChoppedBroccoli])

# R4: Tomato + Potato + Carrot + Broccoli
TomatoPotatoCarrotBroccoliPlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                                             contains=[TomatoPotatoPlate, ChoppedCarrot, ChoppedBroccoli])

# Delivered Salads
LettuceSalad = RecipeNode(root_type=DeliverSquare, id_num=next(id_generator), name="DeliverSquare", conditions=None,
                         contains=[LettuceSaladPlate])
TomatoSalad = RecipeNode(root_type=DeliverSquare, id_num=next(id_generator), name="DeliverSquare", conditions=None,
                         contains=[TomatoSaladPlate])
TomatoLettuceSalad = RecipeNode(root_type=DeliverSquare, id_num=next(id_generator), name="DeliverSquare",
                                conditions=None, contains=[TomatoLettucePlate])
LettuceTomatoSalad = RecipeNode(root_type=DeliverSquare, id_num=next(id_generator), name="DeliverSquare",
                                conditions=None, contains=[LettuceTomatoPlate])
TomatoOnionSalad = RecipeNode(root_type=DeliverSquare, id_num=next(id_generator), name="DeliverSquare",
                              conditions=None, contains=[TomatoOnionPlate])
OnionTomatoSalad = RecipeNode(root_type=DeliverSquare, id_num=next(id_generator), name="DeliverSquare",
                                conditions=None, contains=[OnionTomatoPlate])
TomatoCarrotSalad = RecipeNode(root_type=DeliverSquare, id_num=next(id_generator), name="DeliverSquare",
                                conditions=None, contains=[TomatoCarrotPlate])
CarrotTomatoSalad = RecipeNode(root_type=DeliverSquare, id_num=next(id_generator), name="DeliverSquare",
                                conditions=None, contains=[CarrotTomatoPlate])
LettuceOnionSalad = RecipeNode(root_type=DeliverSquare, id_num=next(id_generator), name="DeliverSquare",
                                conditions=None, contains=[LettuceOnionPlate])
OnionLettuceSalad = RecipeNode(root_type=DeliverSquare, id_num=next(id_generator), name="DeliverSquare",
                                conditions=None, contains=[OnionLettucePlate])
LettuceCarrotSalad = RecipeNode(root_type=DeliverSquare, id_num=next(id_generator), name="DeliverSquare",
                                conditions=None, contains=[LettuceCarrotPlate])
CarrotLettuceSalad = RecipeNode(root_type=DeliverSquare, id_num=next(id_generator), name="DeliverSquare",
                                conditions=None, contains=[CarrotLettucePlate])
OnionCarrotSalad = RecipeNode(root_type=DeliverSquare, id_num=next(id_generator), name="DeliverSquare",
                              conditions=None, contains=[OnionCarrotPlate])
CarrotOnionSalad = RecipeNode(root_type=DeliverSquare, id_num=next(id_generator), name="DeliverSquare",
                                conditions=None, contains=[CarrotOnionPlate])
TomatoPotatoSalad = RecipeNode(root_type=DeliverSquare, id_num=next(id_generator), name="DeliverSquare",
                               conditions=None, contains=[TomatoPotatoPlate])
PotatoTomatoSalad = RecipeNode(root_type=DeliverSquare, id_num=next(id_generator), name="DeliverSquare",
                               conditions=None, contains=[PotatoTomatoPlate])
LettucePotatoSalad = RecipeNode(root_type=DeliverSquare, id_num=next(id_generator), name="DeliverSquare",
                               conditions=None, contains=[LettucePotatoPlate])
PotatoLettuceSalad = RecipeNode(root_type=DeliverSquare, id_num=next(id_generator), name="DeliverSquare",
                               conditions=None, contains=[PotatoLettucePlate])
OnionPotatoSalad = RecipeNode(root_type=DeliverSquare, id_num=next(id_generator), name="DeliverSquare",
                               conditions=None, contains=[OnionPotatoPlate])
PotatoOnionSalad = RecipeNode(root_type=DeliverSquare, id_num=next(id_generator), name="DeliverSquare",
                               conditions=None, contains=[PotatoOnionPlate])
CarrotPotatoSalad = RecipeNode(root_type=DeliverSquare, id_num=next(id_generator), name="DeliverSquare",
                               conditions=None, contains=[CarrotPotatoPlate])
PotatoCarrotSalad = RecipeNode(root_type=DeliverSquare, id_num=next(id_generator), name="DeliverSquare",
                               conditions=None, contains=[PotatoCarrotPlate])
TomatoBroccoliSalad = RecipeNode(root_type=DeliverSquare, id_num=next(id_generator), name="DeliverSquare",
                               conditions=None, contains=[TomatoBroccoliPlate])
BroccoliTomatoSalad = RecipeNode(root_type=DeliverSquare, id_num=next(id_generator), name="DeliverSquare",
                               conditions=None, contains=[BroccoliTomatoPlate])
LettuceBroccoliSalad = RecipeNode(root_type=DeliverSquare, id_num=next(id_generator), name="DeliverSquare",
                               conditions=None, contains=[LettuceBroccoliPlate])
BroccoliLettuceSalad = RecipeNode(root_type=DeliverSquare, id_num=next(id_generator), name="DeliverSquare",
                               conditions=None, contains=[BroccoliLettucePlate])
OnionBroccoliSalad = RecipeNode(root_type=DeliverSquare, id_num=next(id_generator), name="DeliverSquare",
                               conditions=None, contains=[OnionBroccoliPlate])
BroccoliOnionSalad = RecipeNode(root_type=DeliverSquare, id_num=next(id_generator), name="DeliverSquare",
                               conditions=None, contains=[BroccoliOnionPlate])
CarrotBroccoliSalad = RecipeNode(root_type=DeliverSquare, id_num=next(id_generator), name="DeliverSquare",
                               conditions=None, contains=[CarrotBroccoliPlate])
BroccoliCarrotSalad = RecipeNode(root_type=DeliverSquare, id_num=next(id_generator), name="DeliverSquare",
                               conditions=None, contains=[BroccoliCarrotPlate])
PotatoBroccoliSalad = RecipeNode(root_type=DeliverSquare, id_num=next(id_generator), name="DeliverSquare",
                               conditions=None, contains=[PotatoBroccoliPlate])
BroccoliPotatoSalad = RecipeNode(root_type=DeliverSquare, id_num=next(id_generator), name="DeliverSquare",
                               conditions=None, contains=[BroccoliPotatoPlate])

# 4-ingredient salads
LettuceTomatoOnionCarrotSalad = RecipeNode(root_type=DeliverSquare, id_num=next(id_generator), name="DeliverSquare",
                                           conditions=None, contains=[LettuceTomatoOnionCarrotPlate])
LettucePotatoOnionBroccoliSalad = RecipeNode(root_type=DeliverSquare, id_num=next(id_generator), name="DeliverSquare",
                                             conditions=None, contains=[LettucePotatoOnionBroccoliPlate])
TomatoPotatoCarrotBroccoliSalad = RecipeNode(root_type=DeliverSquare, id_num=next(id_generator), name="DeliverSquare",
                                             conditions=None, contains=[TomatoPotatoCarrotBroccoliPlate])

ChoppedOnion = RecipeNode(root_type=DeliverSquare, id_num=next(id_generator), name="DeliverSquare",
                          conditions=None, contains=[ChoppedOnionPlate])
ChoppedCarrot = RecipeNode(root_type=DeliverSquare, id_num=next(id_generator), name="DeliverSquare",
                          conditions=None, contains=[ChoppedCarrotPlate])
ChoppedPotato = RecipeNode(root_type=DeliverSquare, id_num=next(id_generator), name="DeliverSquare",
                           conditions=None, contains=[ChoppedPotatoPlate])
ChoppedBroccoli = RecipeNode(root_type=DeliverSquare, id_num=next(id_generator), name="DeliverSquare",
                           conditions=None, contains=[ChoppedBroccoliPlate])
MashedCarrot = RecipeNode(root_type=DeliverSquare, id_num=next(id_generator), name="DeliverSquare",
                          conditions=None, contains=[MasedCarrotPlate])


# ==============================
# 6-recipe domain additions
# ==============================
# Chopped ingredient nodes (new)
ChoppedCucumber = RecipeNode(root_type=Cucumber, id_num=next(id_generator), name="Cucumber",
                             conditions=[("chop_state", ChopFoodStates.CHOPPED)])
ChoppedPepper = RecipeNode(root_type=Pepper, id_num=next(id_generator), name="Pepper",
                           conditions=[("chop_state", ChopFoodStates.CHOPPED)])
ChoppedSpinach = RecipeNode(root_type=Spinach, id_num=next(id_generator), name="Spinach",
                            conditions=[("chop_state", ChopFoodStates.CHOPPED)])
ChoppedMushroom = RecipeNode(root_type=Mushroom, id_num=next(id_generator), name="Mushroom",
                             conditions=[("chop_state", ChopFoodStates.CHOPPED)])
ChoppedPumpkin = RecipeNode(root_type=Pumpkin, id_num=next(id_generator), name="Pumpkin",
                            conditions=[("chop_state", ChopFoodStates.CHOPPED)])
ChoppedGarlic = RecipeNode(root_type=Garlic, id_num=next(id_generator), name="Garlic",
                           conditions=[("chop_state", ChopFoodStates.CHOPPED)])

# Single-ingredient salad plates
ChoppedCucumberPlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                                  contains=[ChoppedCucumber])
ChoppedPepperPlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                                contains=[ChoppedPepper])
ChoppedSpinachPlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                                 contains=[ChoppedSpinach])
ChoppedMushroomPlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                                  contains=[ChoppedMushroom])
ChoppedPumpkinPlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                                 contains=[ChoppedPumpkin])
ChoppedGarlicPlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                                contains=[ChoppedGarlic])

# 2-ingredient plates: R4 = CucumberPepper (X=Cucumber left, U=Pepper right)
CucumberPepperPlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                                 contains=[ChoppedCucumberPlate, ChoppedPepper])
PepperCucumberPlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                                 contains=[ChoppedPepperPlate, ChoppedCucumber])
# R5 = SpinachMushroom (Y=Spinach left, V=Mushroom right)
SpinachMushroomPlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                                  contains=[ChoppedSpinachPlate, ChoppedMushroom])
MushroomSpinachPlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                                  contains=[ChoppedMushroomPlate, ChoppedSpinach])
# R6 = PumpkinGarlic (Z=Pumpkin left, W=Garlic right)
PumpkinGarlicPlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                                contains=[ChoppedPumpkinPlate, ChoppedGarlic])
GarlicPumpkinPlate = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                                contains=[ChoppedGarlicPlate, ChoppedPumpkin])

# Final delivery nodes
CucumberPepperSalad = RecipeNode(root_type=DeliverSquare, id_num=next(id_generator), name="DeliverSquare",
                                 conditions=None, contains=[CucumberPepperPlate])
PepperCucumberSalad = RecipeNode(root_type=DeliverSquare, id_num=next(id_generator), name="DeliverSquare",
                                 conditions=None, contains=[PepperCucumberPlate])
SpinachMushroomSalad = RecipeNode(root_type=DeliverSquare, id_num=next(id_generator), name="DeliverSquare",
                                  conditions=None, contains=[SpinachMushroomPlate])
MushroomSpinachSalad = RecipeNode(root_type=DeliverSquare, id_num=next(id_generator), name="DeliverSquare",
                                  conditions=None, contains=[MushroomSpinachPlate])
PumpkinGarlicSalad = RecipeNode(root_type=DeliverSquare, id_num=next(id_generator), name="DeliverSquare",
                                conditions=None, contains=[PumpkinGarlicPlate])
GarlicPumpkinSalad = RecipeNode(root_type=DeliverSquare, id_num=next(id_generator), name="DeliverSquare",
                                conditions=None, contains=[GarlicPumpkinPlate])

# ==============================
# 6recipe_x4 domain (2026-10-28): six 4-ingredient recipes on the same 12 ingredients, arranged as the
# octahedron K_{2,2,2}: recipes are vertices, ingredients are edges, so EVERY ingredient belongs to exactly
# two recipes of DIFFERENT types (A/B/C) and any two ingredients of one recipe identify it. Each recipe has
# 2 LEFT-side + 2 RIGHT-side ingredients (map fc_small_divider_6recipe). Plate nodes are ORDER-FREE: the
# full plate node holds four single-ingredient plate nodes, so each plating pays progress in any order.
# (Appending here only grows NUM_GOALS, which feeds info['goal_vector'] and nothing else.)
# ==============================
X4_RECIPES = {   # name: (left ingredients, right ingredients)
    'X4_A1': (('Lettuce', 'Tomato'),   ('Onion', 'Carrot')),
    'X4_A2': (('Potato', 'Cucumber'),  ('Broccoli', 'Pepper')),
    'X4_B1': (('Lettuce', 'Spinach'),  ('Broccoli', 'Mushroom')),
    'X4_B2': (('Potato', 'Pumpkin'),   ('Onion', 'Garlic')),
    'X4_C1': (('Tomato', 'Spinach'),   ('Pepper', 'Garlic')),
    'X4_C2': (('Cucumber', 'Pumpkin'), ('Carrot', 'Mushroom')),
}
# Fresh chopped-ingredient nodes: the legacy names ChoppedOnion/Carrot/Potato/Broccoli are REBOUND further up
# to single-ingredient recipe roots, so they cannot be reused here.
_X4_INGREDIENT_CLASSES = {'Lettuce': Lettuce, 'Tomato': Tomato, 'Onion': Onion, 'Carrot': Carrot,
                          'Potato': Potato, 'Broccoli': Broccoli, 'Cucumber': Cucumber, 'Pepper': Pepper,
                          'Spinach': Spinach, 'Mushroom': Mushroom, 'Pumpkin': Pumpkin, 'Garlic': Garlic}
_X4_CHOPPED = {_ing: RecipeNode(root_type=_cls, id_num=next(id_generator), name=_ing,
                                conditions=[("chop_state", ChopFoodStates.CHOPPED)])
               for _ing, _cls in _X4_INGREDIENT_CLASSES.items()}
X4_ROOTS = {}
for _name, (_left, _right) in X4_RECIPES.items():
    _singles = [RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None,
                           contains=[_X4_CHOPPED[_ing]]) for _ing in _left + _right]
    _full = RecipeNode(root_type=Plate, id_num=next(id_generator), name="Plate", conditions=None, contains=_singles)
    X4_ROOTS[_name] = RecipeNode(root_type=DeliverSquare, id_num=next(id_generator), name="DeliverSquare",
                                 conditions=None, contains=[_full])

# this one increments one further and is thus the amount of ids we have given since
# we started counting at zero.
NUM_GOALS = next(id_generator)

RECIPES = {
    "LettuceSalad":lambda: deepcopy(Recipe(LettuceSalad, name='LettuceSalad')),
    "TomatoSalad": lambda: deepcopy(Recipe(TomatoSalad, name='TomatoSalad')),
    "TomatoLettuceSalad": lambda: deepcopy(Recipe(TomatoLettuceSalad, name='TomatoLettuceSalad')),
    "LettuceTomatoSalad": lambda: deepcopy(Recipe(LettuceTomatoSalad, name='LettuceTomatoSalad')),
    "TomatoOnionSalad": lambda: deepcopy(Recipe(TomatoOnionSalad, name='TomatoOnionSalad')),
    "OnionTomatoSalad": lambda: deepcopy(Recipe(OnionTomatoSalad, name='OnionTomatoSalad')),
    "TomatoCarrotSalad": lambda: deepcopy(Recipe(TomatoCarrotSalad, name='TomatoCarrotSalad')),
    "CarrotTomatoSalad": lambda: deepcopy(Recipe(CarrotTomatoSalad, name='CarrotTomatoSalad')),
    "LettuceOnionSalad": lambda: deepcopy(Recipe(LettuceOnionSalad, name='LettuceOnionSalad')),
    "OnionLettuceSalad": lambda: deepcopy(Recipe(OnionLettuceSalad, name='OnionLettuceSalad')),
    "LettuceCarrotSalad": lambda: deepcopy(Recipe(LettuceCarrotSalad, name='LettuceCarrotSalad')),
    "CarrotLettuceSalad": lambda: deepcopy(Recipe(CarrotLettuceSalad, name='CarrotLettuceSalad')),
    "OnionCarrotSalad": lambda: deepcopy(Recipe(OnionCarrotSalad, name='OnionCarrotSalad')),
    "CarrotOnionSalad": lambda: deepcopy(Recipe(CarrotOnionSalad, name='CarrotOnionSalad')),
    "TomatoPotatoSalad": lambda: deepcopy(Recipe(TomatoPotatoSalad, name='TomatoPotatoSalad')),
    "PotatoTomatoSalad": lambda: deepcopy(Recipe(PotatoTomatoSalad, name='PotatoTomatoSalad')),
    "LettucePotatoSalad": lambda: deepcopy(Recipe(LettucePotatoSalad, name='LettucePotatoSalad')),
    "PotatoLettuceSalad": lambda: deepcopy(Recipe(PotatoLettuceSalad, name='PotatoLettuceSalad')),
    "OnionPotatoSalad": lambda: deepcopy(Recipe(OnionPotatoSalad, name='OnionPotatoSalad')),
    "PotatoOnionSalad": lambda: deepcopy(Recipe(PotatoOnionSalad, name='PotatoOnionSalad')),
    "CarrotPotatoSalad": lambda: deepcopy(Recipe(CarrotPotatoSalad, name='CarrotPotatoSalad')),
    "PotatoCarrotSalad": lambda: deepcopy(Recipe(PotatoCarrotSalad, name='PotatoCarrotSalad')),
    "TomatoBroccoliSalad": lambda: deepcopy(Recipe(TomatoBroccoliSalad, name='TomatoBroccoliSalad')),
    "BroccoliTomatoSalad": lambda: deepcopy(Recipe(BroccoliTomatoSalad, name='BroccoliTomatoSalad')),
    "LettuceBroccoliSalad": lambda: deepcopy(Recipe(LettuceBroccoliSalad, name='LettuceBroccoliSalad')),
    "BroccoliLettuceSalad": lambda: deepcopy(Recipe(BroccoliLettuceSalad, name='BroccoliLettuceSalad')),
    "OnionBroccoliSalad": lambda: deepcopy(Recipe(OnionBroccoliSalad, name='OnionBroccoliSalad')),
    "BroccoliOnionSalad": lambda: deepcopy(Recipe(BroccoliOnionSalad, name='BroccoliOnionSalad')),
    "CarrotBroccoliSalad": lambda: deepcopy(Recipe(CarrotBroccoliSalad, name='CarrotBroccoliSalad')),
    "BroccoliCarrotSalad": lambda: deepcopy(Recipe(BroccoliCarrotSalad, name='BroccoliCarrotSalad')),
    "PotatoBroccoliSalad": lambda: deepcopy(Recipe(PotatoBroccoliSalad, name='PotatoBroccoliSalad')),
    "BroccoliPotatoSalad": lambda: deepcopy(Recipe(BroccoliPotatoSalad, name='BroccoliPotatoSalad')),
    
    # 4-ingredient recipes
    "LettuceTomatoOnionCarrotSalad": lambda: deepcopy(Recipe(LettuceTomatoOnionCarrotSalad, name='LettuceTomatoOnionCarrotSalad')),
    "LettucePotatoOnionBroccoliSalad": lambda: deepcopy(Recipe(LettucePotatoOnionBroccoliSalad, name='LettucePotatoOnionBroccoliSalad')),
    "TomatoPotatoCarrotBroccoliSalad": lambda: deepcopy(Recipe(TomatoPotatoCarrotBroccoliSalad, name='TomatoPotatoCarrotBroccoliSalad')),
    
    # "TomatoLettuceOnionSalad": lambda: deepcopy(Recipe(TomatoLettuceOnionSalad, name='TomatoLettuceOnionSalad')),
    "ChoppedCarrot": lambda: deepcopy(Recipe(ChoppedCarrot, name='ChoppedCarrot')),
    "ChoppedOnion": lambda: deepcopy(Recipe(ChoppedOnion, name='ChoppedOnion')),
    "ChoppedPotato": lambda: deepcopy(Recipe(ChoppedPotato, name='ChoppedPotato')),
    "ChoppedBroccoli": lambda: deepcopy(Recipe(ChoppedBroccoli, name='ChoppedBroccoli')),
    "MashedCarrot": lambda: deepcopy(Recipe(MashedCarrot, name='MashedCarrot')),

    # 6-recipe domain (R4-R6 + reverses)
    "CucumberPepperSalad": lambda: deepcopy(Recipe(CucumberPepperSalad, name='CucumberPepperSalad')),
    "PepperCucumberSalad": lambda: deepcopy(Recipe(PepperCucumberSalad, name='PepperCucumberSalad')),
    "SpinachMushroomSalad": lambda: deepcopy(Recipe(SpinachMushroomSalad, name='SpinachMushroomSalad')),
    "MushroomSpinachSalad": lambda: deepcopy(Recipe(MushroomSpinachSalad, name='MushroomSpinachSalad')),
    "PumpkinGarlicSalad": lambda: deepcopy(Recipe(PumpkinGarlicSalad, name='PumpkinGarlicSalad')),
    "GarlicPumpkinSalad": lambda: deepcopy(Recipe(GarlicPumpkinSalad, name='GarlicPumpkinSalad')),
}
# 6recipe_x4 domain recipes (see X4_RECIPES above)
RECIPES.update({_n: (lambda _r=_r, _n=_n: deepcopy(Recipe(_r, name=_n))) for _n, _r in X4_ROOTS.items()})
