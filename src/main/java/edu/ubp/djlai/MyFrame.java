package edu.ubp.djlai;

import java.awt.EventQueue;
import java.awt.Graphics;
import java.awt.Image;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.border.EmptyBorder;

/**
 *
 * @author agustin
 */
public class MyFrame extends JFrame {

    private JPanel contentPane;
    DjlAi djl;
    VideoCap videoCap;

    /**
     * Launch the application.
     */
    public static void main(String[] args) {
        EventQueue.invokeLater(new Runnable() {
            public void run() {
                try {
                    MyFrame frame = new MyFrame();
                    frame.setVisible(true);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        });
    }

    /**
     * Create the frame.
     */
    public MyFrame() {
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setBounds(100, 100, 650, 490);
        this.contentPane = new JPanel();
        this.contentPane.setBorder(new EmptyBorder(5, 5, 5, 5));
        setContentPane(contentPane);
        this.contentPane.setLayout(null);
        this.djl = new DjlAi();
        this.videoCap = new VideoCap(this.djl);
        new MyThread().start();
    }

    @Override
    public void paint(Graphics g) {
        g = contentPane.getGraphics();
        g.drawImage((Image) videoCap.getOneFrame(), 0, 0, this);
    }

    class MyThread extends Thread {

        @Override
        public void run() {
            for (;;) {
                repaint();
                try {
                    Thread.sleep(30);
                } catch (InterruptedException e) {
                }
            }
        }
    }
}
