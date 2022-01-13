"""
import cv2 as cv
import numpy as np

"""
import threading
from utilities.StandardVideoOperations import *
from timepy import Timer

def main_left(KNNSX,loc_top_sx,loc_bottom_sx):
    global svo
    global startingFrame
    global leftCut
    global cut_flow_top_sx
    global cut_flow_bottom_sx
    global leftResult
    global frame_counter
    global top_frameSX
    #global finalFrame_sx
    # global middle_frameSX
    global contagiu_top_sx
    global contagiu_bottom_sx
    global last_score_frameSX
    global last_score_frameDX
    global score_SX
    global canestri_corretti_sx
    global TPR_SX
    global FPR_SX

    #QUARTO QUARTO:
    
    Frame_canestri_sx = np.array([4747, 6262, 6616, 6440, 8741, 11889, 13972, 14167, 17053,
                                  19532, 19844, 21832, 22644, 23372, 24900, 24945, 28385, 30538,
                                  32469, 32770, 33989, 37446])
    """
    #PRIMO QUARTO:
    Frame_canestri_sx = np.array([1760, 2488, 4370, 4856, 9116, 9715, 11420, 12096, 21745,
                                  23648, 23850, 25623, 27980, 28368, 29041, 35906])

    """

    # posizioni dei vari rettangoli che vengono utilizzati nell’analisi
    upper_left1 = (80, 85)
    bottom_right1 = (140, 95)

    # upper_left2 = (80, 125)
    # bottom_right2 = (140, 135)

    upper_left3 = (70, 160)
    bottom_right3 = (160, 170)

    #leftCut= left_cut
    #leftCut = svo.cut_left(startingFrame)  # ritaglio della RoI
    # filtri mediani non utilizzati perché portavano ad errori
    # blurred = cv.GaussianBlur(leftCut, (5, 5), 0)
    # blurred = cv.medianBlur(blurred, 5)
    #hsvFrame = cv.cvtColor(leftCut, cv.COLOR_BGR2HSV)  # conversione in HSV per poter sogliare meglio sulla base del colore della palla
    #res = svo.get_hsvmask_on_ball(hsvFrame)  # esecuzione della sogliatura del colore con valori trovati usando l'euristica
    #finalFrame = svo.get_knn_on_left_frame(res)  # utilizzo della background subtraction KNN per la parte sinistra
    finalFrame = KNNSX
    leftResult = cv.cvtColor(finalFrame, cv.COLOR_GRAY2BGR)  # conversione in BGR per disegnare i rettangoli colorati in base al rilevamento della palla

    if frame_counter > 10:  # per evitare problemi dovuti alla background subtraction sinistra nei primi frame
        if svo.spotBallOnTop_left(finalFrame):  # ricerchiamo la presenza della palla nel rettangolo sopra il canestro
            top_frameSX = frame_counter  # se e' presente salviamo il numero del frame

            #if loc_top_sx == 0:
            print("Top SX, frame:", top_frameSX) # segnaliamo in output il numero del frame in cui la palla e' stata rilevata
            leftResult = svo.draw_rectangle(leftResult, upper_left1, bottom_right1, "green")  # coloriamo il rettangolo di verde
        else:
            leftResult = svo.draw_rectangle(leftResult, upper_left1, bottom_right1, "red")  # altrimenti lasciamo il rettangolo colorato di rosso

        """
        # il rettangolo centrale a livello della retina e' stato provato ma con risultati peggiori in quanto non veniva sempre segnalata la presenza della palla
        # e l'attivazione era dovuta principalmente al movimento retina o a causa del  monitor dietro il canestro (che spesso influisce e causa problemi anche sul rilevamento della palla nel rettangolo sotto il canestro)
        if svo.spotBallOnMedium_left(finalFrame) and 2 < (frame_counter - top_frameSX) < 15:  # ricerchiamo la presenza della palla nel rettangolo a livello della retina
            middle_frameSX = frame_counter  # se e' presente dopo essere stata rilevata nel rettangolo sopra tra 2 e 15 frame prima salviamo il numero del frame
            print("Middle SX, frame:", middle_frameSX)  # segnaliamo in output il numero del frame in cui la palla e' stata rilevata
            leftResult = svo.draw_rectangle(leftResult, upper_left2, bottom_right2, "green")  # coloriamo il rettangolo di verde
        else:
            leftResult = svo.draw_rectangle(leftResult, upper_left2, bottom_right2, "red")  # altrimenti lasciamo il rettangolo colorato di rosso
        """

        if svo.spotBallOnBottom_left(finalFrame) and 3 < (frame_counter - top_frameSX) < 25 and (frame_counter - last_score_frameSX) > 50 and (frame_counter - last_score_frameDX) > 50:  # ricerchiamo la presenza della palla nel rettangolo sotto il canestro
            if loc_bottom_sx == 0:
                last_score_frameSX = frame_counter  # se e' presente dopo essere stata rilevata nel rettangolo sopra tra 3 e 25 frame prima e l’ultimo score e' stato segnato almeno 50 frame fa salviamo il numero del frame
                score_SX += 1  # incremento del contatore del numero di canetri rilevati a sinistra
                for i in np.nditer(Frame_canestri_sx):
                    if ((last_score_frameSX in range(i, i + 35)) or (last_score_frameSX in range(i - 35, i))):
                        canestri_corretti_sx += 1
                        print("CANESTRO_SX CORRETTAMENTE RILEVATO al frame = ", last_score_frameSX)
                print("Score SX numero", score_SX, "al frame:",
                      last_score_frameSX)  # segnaliamo in output il frame attuale a cui e' stato rilevato il canestro sinistro

            leftResult = svo.draw_rectangle(leftResult, upper_left3, bottom_right3, "green")  # coloriamo il rettangolo di verde
        else:
            leftResult = svo.draw_rectangle(leftResult, upper_left3, bottom_right3, "red")  # altrimenti lasciamo il rettangolo colorato di rosso

        # questa parte e' stata utilizzata per valutare che non venisse segnalata nuovamente la presenza della palla nel rettangolo in alto
        if top_frameSX - last_score_frameSX <  0 and frame_counter - last_score_frameSX == 5:  # effettuiamo un controllo 5 frame dopo che e' stato segnalato il canestro per valutare se la palla e' stata rilevata nuovamente
            print("Score SX numero", score_SX, "con precauzione top")  # se la condizione e' verificata segnaliamo in output che il canestro e' stato segnato con una ulteriore precauzione

    # calcolo TPR & FPR
    TPR_SX = canestri_corretti_sx / Frame_canestri_sx.size
    FPR_SX = (score_SX - canestri_corretti_sx) / Frame_canestri_sx.size

    """
     if (loc_top_sx == 0 and contagiu_top_sx == 0):  # situazione base
        topFrame_sx = frame_counter  # rilevo la palla e segno il frame corrispondente

    if (loc_top_sx == 0):  # situazione base
        contagiu_top_sx += 1

    if (loc_bottom_sx == 0 and (
            frame_counter - top_frameSX > 6)):  # è passato almeno 1/4 secondo da rilevazione sotto e sopra
        contagiu_bottom_sx += 1

    if (frame_counter - top_frameSX > 100):
        contagiu_top_sx = 0
        contagiu_bottom_sx = 0

    if ((contagiu_top_sx > 2) and (
            frame_counter - last_score_frameSX > 75)):  # sono passati 3 secondi dall'ultimo canestro
        last_score_frame_sx = frame_counter
        canestri_sx += 1

    
    """


def main_right(KNNDX,loc_top_dx,loc_bottom_dx):
    global svo
    global startingFrame
    global rightCut
    global cut_flow_top_dx
    global cut_flow_bottom_dx
    global rightResult
    global frame_counter
    global top_frameDX
    # global middle_frameDX
    global last_score_frameDX
    global last_score_frameSX
    global score_DX
    global canestri_corretti_dx
    global contagiu_top_dx
    global contagiu_bottom_dx
    global TPR_DX
    global FPR_DX

    #PRIMO TEMPO:
    """
    Frame_canestri_dx = np.array([14132, 17151, 19768, 20355, 21220, 22277, 30539,
                                  35160, 38511])

    """
    #QUARTO TEMPO:


    Frame_canestri_dx = np.array([1576, 2266, 2882, 3309, 5268, 6968, 9364, 12456, 14688,
                                  15770, 16570, 21170, 23098, 27764, 29347, 30080, 35959,
                                  36527, 37003])


    # posizioni dei vari rettangoli che vengono utilizzati nell’analisi
    upper_left1 = (90, 50)
    bottom_right1 = (150, 60)

    # upper_left2 = (90, 100)
    # bottom_right2 = (150, 110)

    upper_left3 = (75, 160)
    bottom_right3 = (175, 170)

    #rightCut = svo.cut_right(startingFrame)  # ritaglio della RoI
    # filtri mediani non utilizzati perché portavano ad errori
    # blurred = cv.GaussianBlur(rightCut, (5, 5), 0)
    # blurred = cv.medianBlur(blurred, 5)
    #hsvFrame = cv.cvtColor(rightCut, cv.COLOR_BGR2HSV)  # conversione in HSV per poter sogliare meglio sulla base del colore della palla
    #res = svo.get_hsvmask_on_ball(hsvFrame)  # esecuzione della sogliatura del colore con valori trovati usando l'euristica
    #finalFrame = svo.get_knn_on_right_frame(res)  # utilizzo della background subtraction KNN per la parte sinistra
    finalFrame = KNNDX
    rightResult = cv.cvtColor(finalFrame, cv.COLOR_GRAY2BGR)  # conversione in BGR per disegnare i rettangoli colorati in base al rilevamento della palla

    if frame_counter > 10:  # per evitare problemi dovuti alla background subtraction destra nei primi frame
        if svo.spotBallOnTop_right(finalFrame) :  # ricerchiamo la presenza della palla nel rettangolo sopra il canestro
            #if loc_top_dx == 0:
            top_frameDX = frame_counter  # se e' presente salviamo il numero del frame
            print("Top DX, frame:",top_frameDX)  # segnaliamo in output il numero del frame in cui la palla e' stata rilevata
            rightResult = svo.draw_rectangle(rightResult, upper_left1, bottom_right1, "green")  # coloriamo il rettangolo di verde
        else:
            rightResult = svo.draw_rectangle(rightResult, upper_left1, bottom_right1, "red")  # altrimenti lasciamo il rettangolo colorato di rosso

        """
        # il rettangolo centrale a livello della retina e' stato provato ma con risultati peggiori in quanto non veniva sempre segnalata la presenza della palla
        # e l'attivazione era dovuta principalmente al movimento retina o a causa del  monitor dietro il canestro (che spesso influisce e causa problemi anche sul rilevamento della palla nel rettangolo sotto il canestro)
        if svo.spotBallOnMedium_right(finalFrame)  and 2 < (frame_counter - top_frameDX) < 15:  # ricerchiamo la presenza della palla nel rettangolo a livello della retina 
            middle_frameDX = frame_counter  # se e' presente dopo essere stata rilevata nel rettangolo sopra tra 2 e 15 frame prima salviamo il numero del frame
            print("Middle DX, frame:", middle_frameDX)  # segnaliamo in output il numero del frame in cui la palla e' stata rilevata
            rightResult = svo.draw_rectangle(returnFrame, upper_left2, bottom_right2, "green")  # coloriamo il rettangolo di verde
        else:
            rightResult = svo.draw_rectangle(returnFrame, upper_left2, bottom_right2, "red")  # altrimenti lasciamo il rettangolo colorato di rosso
        """
        # se e' presente dopo essere stata rilevata nel rettangolo sopra tra 3 e 25 frame prima e l’ultimo score e' stato segnato almeno 50 frame fa
        if svo.spotBallOnBottom_right(finalFrame) and 3 < (frame_counter - top_frameDX) < 25 and (frame_counter - last_score_frameDX) > 50 and \
                (frame_counter - last_score_frameSX) > 50 :
            # ricerchiamo la presenza della palla nel rettangolo sotto il canestro
            if loc_bottom_dx == 0: #se il movimento è verso il basso
                last_score_frameDX = frame_counter  #salviamo il numero del frame
                score_DX += 1  # incremento del contatore del numero di canetri rilevati a destra
                for i in np.nditer(Frame_canestri_dx):
                    if ((last_score_frameDX in range(i, i + 35)) or (last_score_frameDX in range(i - 35, i))):
                        canestri_corretti_dx += 1
                        print("CANESTRO_DX CORRETTAMENTE RILEVATO al frame = ", last_score_frameDX)
                        file = open("statistiche.txt", "a")
                        file.write("CANESTRO_DX CORRETTAMENTE RILEVATO al frame = " + str(last_score_frameDX))
                        file.close()
                print("Score DX numero", score_DX, "al frame:", last_score_frameDX)  # segnaliamo in output il frame attuale a cui e' stato rilevato il canestro destro
                file = open("statistiche.txt", "a")
                file.write("Score DX numero" + str(score_DX) + "al frame:" + str(last_score_frameDX))
                file.close()
            rightResult = svo.draw_rectangle(rightResult, upper_left3, bottom_right3, "green")  # coloriamo il rettangolo di verde
        else:
            rightResult = svo.draw_rectangle(rightResult, upper_left3, bottom_right3, "red")  # altrimenti lasciamo il rettangolo colorato di rosso

        # questa parte e' stata utilizzata per valutare che non venisse segnalata nuovamente la presenza della palla nel rettangolo in alto
        if top_frameDX - last_score_frameDX <  0 and frame_counter - last_score_frameDX == 5:
            # effettuiamo un controllo 5 frame dopo che e' stato segnalato il canestro per valutare se la palla e' stata rilevata nuovamente
            print("Score DX numero", score_DX, "con precauzione top")
            # se la condizione e' verificata segnaliamo in output che il canestro e' stato segnato con una ulteriore precauzione

    TPR_DX = canestri_corretti_dx / Frame_canestri_dx.size
    FPR_DX = (score_DX - canestri_corretti_dx) / Frame_canestri_dx.size

    """
        if (loc_top_dx == 0 and contagiu_top_dx == 0):  # situazione base
        top_frameDX = frame_counter  # rilevo la palla e segno il frame corrispondente

    if (loc_top_dx == 0):  # situazione base
        contagiu_top_dx += 1

    if (loc_bottom_dx == 0 and (
            frame_counter - top_frameDX > 6)):  # è passato almeno 1/4 secondo da rilevazione sotto e sopra
        contagiu_bottom_dx += 1

    if (frame_counter - top_frameDX > 50):
        contagiu_top_dx = 0
        contagiu_bottom_dx = 0

    if ((contagiu_top_dx > 2 and contagiu_bottom_dx > 2) and (
            frame_counter - last_score_frameDX > 75)):  # sono passati 3 secondi dall'ultimo canestro
        last_score_frame_dx = frame_counter
        print("dx: ", frame_counter)
        canestri_dx += 1
    
    """



if __name__ == "__main__":

    t = Timer()
    t.start()

    svo = StandardVideoOperations()  # istanza della nostra classe per eseguire le varie operazioni
    capture = cv.VideoCapture("/home/dennis/Video/BASKET/quarto_tempo.asf")  # rileva dal percorso fornito il video da analizzare
    path = "/home/renzo/Video/BASKET/quarto_tempo.asf" #copia del percorso da passare al metodo per calibrazionbe iniziale colori
    #svo.display_color_suggestion()
    #color=svo.color_calibration(path) #ricavo i valori per la sogliatura hsv

    color_lower = np.array([160, 75, 85])
    color_upper = np.array([180, 255, 255])
    color = (color_lower,color_upper)

    # Array per la memorizzazione dei movimenti
    directions_map_top_sx = np.zeros([args["size"], 5])
    directions_map_bottom_sx = np.zeros([args["size"], 5])

    directions_map_top_dx = np.zeros([args["size"], 5])
    directions_map_bottom_dx = np.zeros([args["size"], 5])

    """
    0	    0	    0	    0	    0       *
    0	    0	    0	    0	    0       |
    0	    0	    0	    0	    0       |
    0	    0	    0	    0	    0       |
    0	    0	    0	    0	    0       |
    0	    0	    0	    0	    0       |
    0	    0	    0	    0	    0	    |
    0	    0	    0	    0	    0	    *
    down	right	up	    left	wait    size
    """


    # dato lo spostamento della videocamera le coordinate cambiano il base al quarto di gioco preso in analisi
    # da impostare per il 1° quarto
    #svo.set_left((455, 955), (655, 1155))
    #svo.set_right((3145, 895), (3345, 1095))

    # da impostare per il 2° quarto
    # svo.set_left((485, 950), (685, 1150))
    # svo.set_right((3185, 900), (3385, 1100))

    # da impostare per il 3° e il 4° quarto
    svo.set_left((540, 940), (740, 1140))
    svo.set_right((3225, 900), (3425, 1100))

    # da impostare per il 4° quarto
    for i in range(-20, 20,10):
        LEFT_upper_left_x_top = 590 +i
        LEFT_upper_left_y_top = 1000 +i
        LEFT_bottom_right_x_top = 690+i
        LEFT_bottom_right_y_top = 1100 +i

        LEFT_upper_left_x_bottom = 590+i
        LEFT_upper_left_y_bottom = 1010 +i
        LEFT_bottom_right_x_bottom = 690+i
        LEFT_bottom_right_y_bottom = 1110 +i
        print("ROI_LEFT_TOP -->" + "(" + str(LEFT_upper_left_x_top) + "," + str(LEFT_upper_left_y_top) + ")" + "x" +
              "(" + str(LEFT_bottom_right_x_top) + "," + str(LEFT_bottom_right_y_top) + ")")

        print("ROI_LEFT_BOTTOM -->" + "(" + str(LEFT_upper_left_x_bottom) + "," + str(
            LEFT_upper_left_y_bottom) + ")" + "x" +
              "(" + str(LEFT_bottom_right_x_bottom) + "," + str(LEFT_bottom_right_y_bottom) + ")")

        file = open("statistiche.txt", "a")
        file.write("\n ")
        file.write("ROI_LEFT_TOP -->" + "(" + str(LEFT_upper_left_x_top) + "," + str(LEFT_upper_left_y_top) + ")" + "x" +
              "(" + str(LEFT_bottom_right_x_top) + "," + str(LEFT_bottom_right_y_top) + ")")
        file.write("\n ")
        file.write("ROI_LEFT_BOTTOM -->" + "(" + str(LEFT_upper_left_x_bottom) + "," + str(
            LEFT_upper_left_y_bottom) + ")" + "x" +
              "(" + str(LEFT_bottom_right_x_bottom) + "," + str(LEFT_bottom_right_y_bottom) + ")")
        file.close()

        svo.set_left_flow((LEFT_upper_left_x_top, LEFT_upper_left_y_top),
                          (LEFT_bottom_right_x_top, LEFT_bottom_right_y_top),
                          (LEFT_upper_left_x_bottom, LEFT_upper_left_y_bottom),
                          (LEFT_bottom_right_x_bottom, LEFT_bottom_right_y_bottom))

        RIGHT_upper_left_x_top = 3320
        RIGHT_upper_left_y_top = 950
        RIGHT_bottom_right_x_top = 3420
        RIGHT_bottom_right_y_top = 1050

        RIGHT_upper_left_x_bottom = 3320
        RIGHT_upper_left_y_bottom = 980
        RIGHT_bottom_right_x_bottom = 3420
        RIGHT_bottom_right_y_bottom = 1080
        print("ROI_RIGHT_TOP -->" + "(" + str(RIGHT_upper_left_x_top) + "," + str(RIGHT_upper_left_y_top) + ")" + "x" +
              "(" + str(RIGHT_bottom_right_x_top) + "," + str(RIGHT_bottom_right_y_top) + ")")

        print("ROI_RIGHT_BOTTOM -->" + "(" + str(RIGHT_upper_left_x_bottom) + "," + str(
            RIGHT_upper_left_y_bottom) + ")" + "x" +
              "(" + str(RIGHT_bottom_right_x_bottom) + "," + str(RIGHT_bottom_right_y_bottom) + ")")

        file = open("statistiche.txt", "a")
        file.write("ROI_RIGHT_TOP -->" + "(" + str(RIGHT_upper_left_x_top) + "," + str(RIGHT_upper_left_y_top) + ")" + "x" +
              "(" + str(RIGHT_bottom_right_x_top) + "," + str(RIGHT_bottom_right_y_top) + ")")
        file.write("\n ")
        file.write("ROI_RIGHT_BOTTOM -->" + "(" + str(RIGHT_upper_left_x_bottom) + "," + str(
            RIGHT_upper_left_y_bottom) + ")" + "x" +
              "(" + str(RIGHT_bottom_right_x_bottom) + "," + str(RIGHT_bottom_right_y_bottom) + ")")
        file.close()

        svo.set_right_flow((RIGHT_upper_left_x_top, RIGHT_upper_left_y_top),
                           (RIGHT_bottom_right_x_top, RIGHT_bottom_right_y_top),
                           (RIGHT_upper_left_x_bottom, RIGHT_upper_left_y_bottom),
                           (RIGHT_bottom_right_x_bottom, RIGHT_bottom_right_y_bottom))

        """
        # da impostare per il 1° quarto
    
        #for i in range(-1, 1):
        i=0
        LEFT_upper_left_x_top = 525 + i
        LEFT_upper_left_y_top = 985 + i
        LEFT_bottom_right_x_top = 605 + i
        LEFT_bottom_right_y_top = 1075 + i
    
        LEFT_upper_left_x_bottom = 525 + i
        LEFT_upper_left_y_bottom = 1065 + i
        LEFT_bottom_right_x_bottom = 605 + i
        LEFT_bottom_right_y_bottom = 1155 + i
        print("ROI_LEFT_top -->" + "(" + str(LEFT_upper_left_x_top) + "," + str(LEFT_upper_left_y_top) + ")" + "x" +
              "(" + str(LEFT_bottom_right_x_top) + "," + str(LEFT_bottom_right_y_top) + ")")
        print(
            "ROI_LEFT_bottom -->" + "(" + str(LEFT_upper_left_x_bottom) + "," + str(LEFT_upper_left_y_bottom) + ")" + "x" +
            "(" + str(LEFT_bottom_right_x_bottom) + "," + str(LEFT_bottom_right_y_bottom) + ")")
        """
        """
        file = open("statistiche.txt", "a")
        file.write("\n ")
        file.write("\nROI_LEFT: " + "(" + str(LEFT_upper_left_x_top) + "," + str(LEFT_upper_left_y_top) + ")" + "x" +
                       "(" + str(LEFT_bottom_right_x_top) + "," + str(LEFT_bottom_right_y_top) + ")")
        file.close()
        """
        """
        svo.set_left_flow((LEFT_upper_left_x_top, LEFT_upper_left_y_top),
                              (LEFT_bottom_right_x_top, LEFT_bottom_right_y_top),
                              (LEFT_upper_left_x_bottom, LEFT_upper_left_y_bottom),
                              (LEFT_bottom_right_x_bottom, LEFT_bottom_right_y_bottom))
    
        RIGHT_upper_left_x_top = 3235 + i
        RIGHT_upper_left_y_top = 905 + i
        RIGHT_bottom_right_x_top = 3335 + i
        RIGHT_bottom_right_y_top = 995 + i
    
        RIGHT_upper_left_x_bottom = 3235 + i
        RIGHT_upper_left_y_bottom = 995 + i
        RIGHT_bottom_right_x_bottom = 3335 + i
        RIGHT_bottom_right_y_bottom = 1085 + i
    
        print("ROI_RIGHT_up -->" + "(" + str(RIGHT_upper_left_x_top) + "," + str(RIGHT_upper_left_y_top) + ")" + "x" +
              "(" + str(RIGHT_bottom_right_x_top) + "," + str(RIGHT_bottom_right_y_top) + ")")
        print("ROI_RIGHT_bottom -->" + "(" + str(RIGHT_upper_left_x_bottom) + "," + str(
            RIGHT_upper_left_y_bottom) + ")" + "x" +
              "(" + str(RIGHT_bottom_right_x_bottom) + "," + str(RIGHT_bottom_right_y_bottom) + ")")
    
        """
        """
        file = open("statistiche.txt", "a")
        file.write("\n ")
        file.write("\nROI_RIGHT: " + "(" + str(RIGHT_upper_left_x_top) + "," + str(RIGHT_upper_left_y_top) + ")" + "x" +
                       "(" + str(RIGHT_bottom_right_x_top) + "," + str(RIGHT_bottom_right_y_top) + ")")
        file.write("\n ")
        file.close()
        
        Frame_canestri_sx = np.array([1760, 2488, 4370, 4856, 9116, 9715, 11420, 12096, 21745,
                                          23648, 23850, 25623, 27980, 28368, 29041, 35906])
    
        Frame_canestri_dx = np.array([14132, 17151, 19768, 20355, 21220, 22277, 30539,
                                          35160, 38511])
        
    
        svo.set_right_flow((RIGHT_upper_left_x_top, RIGHT_upper_left_y_top),
                               (RIGHT_bottom_right_x_top, RIGHT_bottom_right_y_top),
                               (RIGHT_upper_left_x_bottom, RIGHT_upper_left_y_bottom),
                               (RIGHT_bottom_right_x_bottom, RIGHT_bottom_right_y_bottom))
    
        """


        if not capture.isOpened:  # verifica la corretta apertura del video
            print("Unable to open")
            exit(0)

        # indicatori dei frame che saranno utili nell’analisi
        frame_counter = 0
        top_frameSX = 0
        top_frameDX = 0
        # middle_frameSX = 0
        # middle_frameDX = 0
        last_score_frameSX = 0
        last_score_frameDX = 0
        score_DX = 0
        score_SX = 0
        contagiu_top_dx = 0
        contagiu_bottom_dx = 0
        contagiu_top_sx = 0
        contagiu_bottom_sx = 0
        last_score_frame_dx = 0
        canestri_dx = 0
        canestri_corretti_dx = 0
        canestri_sx = 0
        canestri_corretti_sx = 0
        TPR_SX = 0
        TPR_DX = 0
        FPR_DX = 0
        FPR_SX = 0
        #PREPARAZIONE OPTICAL FLOW
        ret, first_frame = capture.read()
        first_cut_frame = svo.cut_frame(first_frame)
        first_cut_flow_top = svo.cut_top_flow(first_frame)
        first_cut_flow_bottom = svo.cut_bottom_flow(first_frame)
        hsvFrame = svo.hsv_thresholding(first_cut_frame,color)

        cut_flow_top = svo.cut_top_flow(first_frame)
        cut_flow_bottom = svo.cut_bottom_flow(first_frame)
        cut_flow_top_hsv = svo.hsv_thresholding(cut_flow_top,color)
        cut_flow_bottom_hsv = svo.hsv_thresholding(cut_flow_bottom, color)
        cut_flow_top_bgr = svo.change_color_space(cut_flow_top_hsv,1)
        cut_flow_bottom_bgr = svo.change_color_space(cut_flow_bottom_hsv, 1)
        gray_previous_top = svo.change_color_space(cut_flow_top_bgr,0)
        gray_previous_bottom = svo.change_color_space(cut_flow_bottom_bgr,0)

        hsv_top_sx = np.zeros_like(first_cut_flow_top[0])
        hsv_top_sx[:, :, 1] = 255
        hsv_bottom_sx = np.zeros_like(first_cut_flow_bottom[0])
        hsv_bottom_sx[:, :, 1] = 255
        hsv_top_dx = np.zeros_like(first_cut_flow_top[1])
        hsv_top_dx[:, :, 1] = 255
        hsv_bottom_dx = np.zeros_like(first_cut_flow_bottom[1])
        hsv_bottom_dx[:, :, 1] = 255

        while True:
            captureStatus, startingFrame = capture.read()  # lettura di un frame del video
            if startingFrame is None:  # interruzione del ciclo alla conclusione del video
                break

            frame_counter += 1  #incremento per avere un indicatore del numero di frame corrente

            # copia del frame del video per eseguire le operazioni nei thread e alla fine riportare i risultati

            cut = svo.cut_frame(startingFrame.copy())
            cutsx = cut[0]
            cutdx = cut[1]

            cut_flow_top = svo.cut_top_flow(startingFrame.copy())
            cut_flow_bottom = svo.cut_bottom_flow(startingFrame.copy())

            cut_flow_top_sx = cut_flow_top[0]
            cut_flow_bottom_sx = cut_flow_bottom[0]
            cut_flow_top_dx = cut_flow_top[1]
            cut_flow_bottom_dx = cut_flow_bottom[1]

            cut_flow_top_hsv = svo.hsv_thresholding(cut_flow_top, color)
            cut_flow_bottom_hsv = svo.hsv_thresholding(cut_flow_bottom, color)
            cut_flow_top_bgr = svo.change_color_space(cut_flow_top_hsv, 1)
            cut_flow_bottom_bgr = svo.change_color_space(cut_flow_bottom_hsv, 1)
            gray_top = svo.change_color_space(cut_flow_top_bgr, 0)
            gray_bottom = svo.change_color_space(cut_flow_bottom_bgr, 0)

            leftCut = cutsx
            rightCut = cutdx
            leftResult = cutsx
            rightResult = cutdx

            cut_hsv = svo.hsv_thresholding(cut,color)
            cut_knn = svo.get_knn_on_frame(cut_hsv)

            #TEST OPTICAL FLOW
            frame2 = cut
            #gray = svo.change_color_space(cut)
            """
            masks = svo.cumpute_denseOpticalFlow(prev_gray,gray,masksx,maskdx)
            rgbsx = cv.cvtColor(masks[0], cv.COLOR_HSV2BGR)
            rgbdx = cv.cvtColor(masks[1], cv.COLOR_HSV2BGR)
            """
            mags_angs = svo.compute_optical_flow(gray_previous_top, gray_previous_bottom, gray_top, gray_bottom)
            # Updates previous frame
            gray_previous_top = gray_top
            gray_previous_bottom = gray_bottom

            ang_top = mags_angs[0]
            mag_top = mags_angs[1]
            ang_bottom = mags_angs[2]
            mag_bottom = mags_angs[3]

            ang_180_top_dx = ang_top[3]
            ang_180_bottom_dx = ang_bottom[3]
            ang_180_top_sx = ang_top[2]
            ang_180_bottom_sx = ang_bottom[2]

            # calcolo la moda degli angoli in un frame determinando l'angolo più frequente
            move_mode_sx = svo.get_movement(ang_top, ang_bottom, mag_top, mag_bottom, 0)
            move_mode_dx = svo.get_movement(ang_top, ang_bottom, mag_top, mag_bottom, 1)

            # riempio i due array stabilendo così il movimento predominante
            directions_map_top_sx, directions_map_bottom_sx = svo.calc_predominant_movement(move_mode_sx,
                                                                                                 directions_map_top_sx,
                                                                                                 directions_map_bottom_sx)
            directions_map_top_dx, directions_map_bottom_dx = svo.calc_predominant_movement(move_mode_dx,
                                                                                                 directions_map_top_dx,
                                                                                                 directions_map_bottom_dx)
            hsv_top_sx[:, :, 0] = ang_180_top_sx
            hsv_top_sx[:, :, 2] = cv.normalize(mag_top[0], None, 0, 255, cv.NORM_MINMAX)
            rgb_top_sx = cv.cvtColor(hsv_top_sx, cv.COLOR_HSV2BGR)

            hsv_bottom_sx[:, :, 0] = ang_180_bottom_sx
            hsv_bottom_sx[:, :, 2] = cv.normalize(mag_bottom[0], None, 0, 255, cv.NORM_MINMAX)
            rgb_bottom_sx = cv.cvtColor(hsv_bottom_sx, cv.COLOR_HSV2BGR)

            hsv_top_dx[:, :, 0] = ang_180_top_dx
            hsv_top_dx[:, :, 2] = cv.normalize(mag_top[1], None, 0, 255, cv.NORM_MINMAX)
            rgb_top_dx = cv.cvtColor(hsv_top_dx, cv.COLOR_HSV2BGR)

            hsv_bottom_dx[:, :, 0] = ang_180_bottom_dx
            hsv_bottom_dx[:, :, 2] = cv.normalize(mag_bottom[1], None, 0, 255, cv.NORM_MINMAX)
            rgb_bottom_dx = cv.cvtColor(hsv_bottom_dx, cv.COLOR_HSV2BGR)

            # CALCOLO CANESTRI SX:
            loc_top_sx = directions_map_top_sx.mean(axis=0).argmax()
            loc_bottom_sx = directions_map_bottom_sx.mean(axis=0).argmax()
            #lancio thread

            # CALCOLO CANESTRI DX:
            loc_top_dx = directions_map_top_dx.mean(axis=0).argmax()
            loc_bottom_dx = directions_map_bottom_dx.mean(axis=0).argmax()
            #lancio thread

            # inizio processing del frame per la parte sinistra e destra in parallelo su due thread
            thread_left = threading.Thread(target=main_left,args=[cut_knn[0],loc_top_sx,loc_bottom_sx])
            thread_left.start()

            thread_right = threading.Thread(target=main_right,args=[cut_knn[1],loc_top_dx,loc_bottom_dx])
            thread_right.start()

            # attendo la fine del processing sul frame e riportiamo a monitor i sultati sulla RoI dopo l’analisi e anche del video originale per visualizzare come l’algoritmo sta svolgendo il proprio lavoro
            thread_left.join()
            cv.imshow("originalFrameSX", cutsx)
            cv.imshow("returnFrameSX", leftResult)
            cv.imshow("top_SX", cut_flow_top_sx)
            cv.imshow("bottom_SX", cut_flow_bottom_sx)

            thread_right.join()
            cv.imshow("originalFrameDX", cutdx)
            cv.imshow("returnFrameDX", rightResult)
            cv.imshow("top_DX", cut_flow_top_dx)
            cv.imshow("bottom_DX", cut_flow_bottom_dx)
            #cv.imshow('SX_UP Mask ', rgb_top_sx)

            key = cv.waitKey(1)  # attesa minima tra un frame e il successivo, ma può essere prolungata per visualizzare più lentamente l’esecuzione dell’algoritmo e capire la cause di eventuali problematiche
            if key == 27:  # per ogni evenienza se viene premuto esc si interrompe l’esecuzione
                break

        print("Score SX effettuati", score_SX)  # riportiamo il numero complessivo di canestri rilevati a sinistra
        print("Score DX effettuati", score_DX)  # riportiamo il numero complessivo di canestri rilevati a destra
        print("TPR_SX: ", TPR_SX)
        print("TPR_DX: ", TPR_DX)
        print("FPR_DX: ", FPR_DX)
        print("FPR_SX: ", FPR_SX)

        cv.destroyAllWindows()  # alla fine dell’esecuzione rimuove tutte le finestre dei frame dallo schermo

        t.stop()
        print("tempo totale esecuzione programma [s]: ",t.total_time)

