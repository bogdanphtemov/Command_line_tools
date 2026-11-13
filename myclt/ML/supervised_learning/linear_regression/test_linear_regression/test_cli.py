import pytest
from unittest.mock import patch, MagicMock, call
from typing import List

from myclt.ML.supervised_learning.linear_regression.cli import main
from myclt.ML.supervised_learning.linear_regression.app_state import AppState


class TestCLIMain:
    """Test cases for the CLI main function."""

    def test_main_exit_immediately(self, monkeypatch):
        """Test that main() exits immediately when Exit is chosen."""
        # Mock ask_choice to return 6 (Exit) immediately
        mock_state = MagicMock(spec=AppState)
        
        with patch('myclt.ML.supervised_learning.linear_regression.cli.AppState', return_value=mock_state):
            with patch('myclt.ML.supervised_learning.linear_regression.cli.ask_choice') as mock_ask_choice:
                with patch('myclt.ML.supervised_learning.linear_regression.cli.clear_screen'):
                    with patch('myclt.ML.supervised_learning.linear_regression.cli.print_header'):
                        with patch('myclt.ML.supervised_learning.linear_regression.cli.print_status'):
                            # Mock ask_choice to choose Exit (index 6) immediately
                            mock_ask_choice.return_value = 6
                            
                            # Call main - it should exit immediately without calling any menu functions
                            result = main()
                            
                            # Should return None (normal exit)
                            assert result is None
                            
                            # Verify ask_choice was called once with correct options
                            mock_ask_choice.assert_called_once_with("", [
                                "Data", "Train", "Evaluate", "Predict", "Visualize", "Save/Load", "Exit"
                            ])

    def test_main_calls_menu_data(self, monkeypatch):
        """Test that main() calls menu_data when Data is chosen."""
        mock_state = MagicMock(spec=AppState)
        
        with patch('myclt.ML.supervised_learning.linear_regression.cli.AppState', return_value=mock_state):
            with patch('myclt.ML.supervised_learning.linear_regression.cli.ask_choice') as mock_ask_choice:
                with patch('myclt.ML.supervised_learning.linear_regression.cli.clear_screen'):
                    with patch('myclt.ML.supervised_learning.linear_regression.cli.print_header'):
                        with patch('myclt.ML.supervised_learning.linear_regression.cli.print_status'):
                            with patch('myclt.ML.supervised_learning.linear_regression.cli.menu_data') as mock_menu_data:
                                # Mock ask_choice to choose Data (0), then Exit (6)
                                mock_ask_choice.side_effect = [0, 6]
                                
                                # Call main
                                result = main()
                                
                                # Should return None (normal exit)
                                assert result is None
                                
                                # Verify menu_data was called
                                mock_menu_data.assert_called_once_with(mock_state)
                                
                                # Verify ask_choice was called twice
                                assert mock_ask_choice.call_count == 2

    def test_main_keyboard_interrupt_during_choice(self, monkeypatch):
        """Test that KeyboardInterrupt during ask_choice results in graceful exit."""
        mock_state = MagicMock(spec=AppState)
        
        with patch('myclt.ML.supervised_learning.linear_regression.cli.AppState', return_value=mock_state):
            with patch('myclt.ML.supervised_learning.linear_regression.cli.ask_choice') as mock_ask_choice:
                with patch('myclt.ML.supervised_learning.linear_regression.cli.clear_screen'):
                    with patch('myclt.ML.supervised_learning.linear_regression.cli.print_header'):
                        with patch('myclt.ML.supervised_learning.linear_regression.cli.print_status'):
                            # Mock ask_choice to raise KeyboardInterrupt
                            mock_ask_choice.side_effect = KeyboardInterrupt()
                            
                            with patch('builtins.print') as mock_print:
                                # Call main
                                result = main()
                                
                                # Should return None (graceful exit)
                                assert result is None
                                
                                # Verify the interruption message was printed
                                mock_print.assert_called_with("\n\nInterrupted by user. Exiting.")

    def test_main_eof_error_during_choice(self, monkeypatch):
        """Test that EOFError during ask_choice results in graceful exit."""
        mock_state = MagicMock(spec=AppState)
        
        with patch('myclt.ML.supervised_learning.linear_regression.cli.AppState', return_value=mock_state):
            with patch('myclt.ML.supervised_learning.linear_regression.cli.ask_choice') as mock_ask_choice:
                with patch('myclt.ML.supervised_learning.linear_regression.cli.clear_screen'):
                    with patch('myclt.ML.supervised_learning.linear_regression.cli.print_header'):
                        with patch('myclt.ML.supervised_learning.linear_regression.cli.print_status'):
                            # Mock ask_choice to raise EOFError
                            mock_ask_choice.side_effect = EOFError()
                            
                            with patch('builtins.print') as mock_print:
                                # Call main
                                result = main()
                                
                                # Should return None (graceful exit)
                                assert result is None
                                
                                # Verify the interruption message was printed
                                mock_print.assert_called_with("\n\nInterrupted by user. Exiting.")

    def test_main_keyboard_interrupt_during_menu_execution(self, monkeypatch):
        """Test that KeyboardInterrupt during menu execution results in graceful exit."""
        mock_state = MagicMock(spec=AppState)
        
        def mock_menu_data_raises_interrupt(state):
            raise KeyboardInterrupt()
        
        with patch('myclt.ML.supervised_learning.linear_regression.cli.AppState', return_value=mock_state):
            with patch('myclt.ML.supervised_learning.linear_regression.cli.ask_choice') as mock_ask_choice:
                with patch('myclt.ML.supervised_learning.linear_regression.cli.clear_screen'):
                    with patch('myclt.ML.supervised_learning.linear_regression.cli.print_header'):
                        with patch('myclt.ML.supervised_learning.linear_regression.cli.print_status'):
                            with patch('myclt.ML.supervised_learning.linear_regression.cli.menu_data', side_effect=mock_menu_data_raises_interrupt):
                                # Mock ask_choice to choose Data (index 0)
                                mock_ask_choice.side_effect = [0]
                                
                                with patch('builtins.print') as mock_print:
                                    # Call main
                                    result = main()
                                    
                                    # Should return None (graceful exit)
                                    assert result is None
                                    
                                    # Verify the interruption message was printed
                                    mock_print.assert_called_with("\n\nInterrupted by user. Exiting.")

    def test_main_multiple_menu_navigation(self, monkeypatch):
        """Test navigating through multiple menus before exiting."""
        mock_state = MagicMock(spec=AppState)
        
        # Track menu calls
        menu_calls = []
        
        def create_menu_mock(name):
            def mock_menu(state):
                menu_calls.append(name)
            return mock_menu
        
        with patch('myclt.ML.supervised_learning.linear_regression.cli.AppState', return_value=mock_state):
            with patch('myclt.ML.supervised_learning.linear_regression.cli.ask_choice') as mock_ask_choice:
                with patch('myclt.ML.supervised_learning.linear_regression.cli.clear_screen'):
                    with patch('myclt.ML.supervised_learning.linear_regression.cli.print_header'):
                        with patch('myclt.ML.supervised_learning.linear_regression.cli.print_status'):
                            with patch('myclt.ML.supervised_learning.linear_regression.cli.menu_data', create_menu_mock("menu_data")):
                                with patch('myclt.ML.supervised_learning.linear_regression.cli.menu_evaluate', create_menu_mock("menu_evaluate")):
                                    with patch('myclt.ML.supervised_learning.linear_regression.cli.menu_predict', create_menu_mock("menu_predict")):
                                        # Navigate: Data -> Evaluate -> Predict -> Exit
                                        mock_ask_choice.side_effect = [0, 2, 3, 6]
                                        
                                        # Call main
                                        result = main()
                                        
                                        # Should return None (normal exit)
                                        assert result is None
                                        
                                        # Verify menu functions were called in correct order
                                        assert menu_calls == ["menu_data", "menu_evaluate", "menu_predict"]
                                        
                                        # Verify ask_choice was called 4 times
                                        assert mock_ask_choice.call_count == 4